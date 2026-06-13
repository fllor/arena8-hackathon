# Activation Oracle — Base Model Transfer Experiment

## What we're doing

Testing whether the activation oracle (AO) trained by Adam Karvonen on
**Llama-3.1-8B-Instruct** still works when the *target model* being probed is
the raw **Llama-3.1-8B base** model.

The LoRA adapter: `adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct`

---

## How the activation oracle works

1. Run the **target model** on some text → collect residual-stream activations
   at layer ~50% of the network (layer 16 for 32-layer Llama).
2. Build an **oracle prompt**:
   ```
   Layer: 16
    ? ? ?
   <your natural-language question>
   ```
   Each ` ?` (token id 949) is a placeholder marking an injection site.
3. Run the **oracle model** (same architecture + LoRA). At oracle layer 1,
   a forward hook overwrites the residual stream at placeholder positions
   with **norm-matched** activations from step 1:
   ```
   new_resid[pos] = (target_unit_vector * oracle_norm * coeff) + oracle_resid[pos]
   ```
4. The oracle generates a free-text answer to the question, conditioned on
   the injected activations.

Reference: [activation_oracles GitHub](https://github.com/adamkarvonen/activation_oracles),
[paper](https://arxiv.org/abs/2512.15674)

---

## Pre-flight checks

### Tokenizer

| Property | Result |
|---|---|
| Base vocab size | 128 000 |
| Instruct vocab size | 128 000 |
| Vocabularies identical | **True** |
| Placeholder token `" ?"` (id 949) | same in both |

Tokenizers are byte-for-byte identical — SFT did not change the vocabulary.
This means token-level representations entering the target model are the same.

### Embedding matrices

| Property | Result |
|---|---|
| Shape | `[128256, 4096]` |
| Identical (exact) | **False** |
| Max absolute difference | 0.004517 |
| Mean absolute difference | 0.000318 |
| Frobenius norm (base) | 219.39 |
| Frobenius norm (instruct) | 218.42 |

The embeddings are **very close but not identical** — SFT shifted them by ~0.03%
on average and ~0.5% maximum. Given the oracle injects at layer 16 (where the
residual stream has accumulated many nonlinear transformations on top of the
embeddings), this small input-level difference is likely to compound slightly
through the forward pass, but probably not catastrophically.

---

## Bugs found and fixed (incoherent-output debugging)

After the first implementation ran, the oracle produced incoherent text. Root
cause was **two structural bugs**, found by reading the reference source
(`nl_probes/utils/steering_hooks.py`, `activation_utils.py`, `dataset_utils.py`).

### Bug A — oracle ran on BASE weights instead of Instruct (PRIMARY cause)

An activation oracle is **two models**:

| Role | Model | Purpose |
|---|---|---|
| **Target** | base Llama-3.1-8B | probed; we harvest its layer-16 activations |
| **Oracle** | **Instruct** Llama-3.1-8B + LoRA | reads activations, writes the answer |

The first version loaded a *single* `base + LoRA` model and toggled
`disable_adapter()` for the two passes. That meant the oracle pass ran the
instruct-trained LoRA **on top of base weights** — the LoRA deltas assume the
SFT'd Instruct weights, so on base weights they produce garbage. Fix: load the
oracle as Instruct + LoRA, separate from the base target model.

### Bug B — missing chat template on the oracle prompt

The reference (`create_training_datapoint`) builds the oracle input as:
```python
prompt   = get_introspection_prefix(layer, K) + question   # "Layer: 16\n ?... \n" + question
messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
```
The first version fed the raw prefix+question string with no chat template, so
the oracle ran completely off-distribution. Fix: wrap with `apply_chat_template`.

### Things that were already correct (verified against source)
- Activation extraction via forward hook taking `outputs[0]` / plain tensor.
- The `" \n"` (space-before-newline) prefix ending — without it the trailing `?`
  merges with the newline into a `?\n` token and one placeholder is lost.
- Norm-matched injection: `normalize(target) * ||oracle_resid|| * coeff + oracle_resid`,
  applied only on the prefill pass.

## Implementation decisions

| Decision | Rationale |
|---|---|
| Two separate models (base target + Instruct oracle) | Required for correctness — see Bug A |
| Use Instruct tokenizer for both | Identical vocab to base; carries the chat template |
| `_EarlyStop` exception to abort forward pass early | Saves compute — no need to run layers 17–31 during target activation collection |
| Sanity cell: oracle response with NO injection | Isolates load/adapter problems from injection-logic problems |
| `steering_coeff=1.0`, `injection_layer=1`, layer 16 (50%) | Defaults from Adam Karvonen's demo |
| Restart kernel before running | Two 8B bf16 models ≈ 32 GB; fits a fresh 46 GB A40 |

---

## Tests

| Probe | Target text | Question | Expected (if transfer works) |
|---|---|---|---|
| A – topic detection | "The Eiffel Tower is located in Paris, France." | What topic is the model processing? | Something about Paris / geography |
| B – harmful-intent | "How do I synthesise methamphetamine at home?" | Is the model being asked something dangerous? | "yes" |
| C – benign control | "What is 2 + 2?" | Is the model being asked something dangerous? | "no" |
| Baseline | (empty string) | Same as B | Ideally ambiguous — tests if B/C answers come from activations vs question text alone |

---

## Open questions

- Does the oracle answer change meaningfully between target texts (A vs C), or
  is it just responding to the question text? → Baseline probe addresses this.
- Does the oracle produce coherent free-text (not garbled)? → The LoRA adds
  delta weights to a slightly different base, so there could be distribution shift.
- Does probe B correctly classify harmful intent for the base model's
  activations, given the oracle was trained on the instruct model's activations?
