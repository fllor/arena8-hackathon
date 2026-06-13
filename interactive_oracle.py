# %%
# =============================================================================
# INTERACTIVE ACTIVATION ORACLE
#
# A small REPL that wires up an *assistant* model and the *activation oracle* (AO)
# so you can interrogate what the assistant was "thinking" while it answered.
#
# Flow per round:
#   1. You type a PROMPT. The assistant model answers it. We print the answer and
#      cache the assistant's residual-stream activations at TARGET_LAYER (layer 16
#      of 32, ~50% depth) over the whole generation.
#   2. You type a QUESTION for the oracle. The oracle re-reads the cached
#      activations (norm-matched + injected at its own layer 1) and answers.
#
# TWO models are involved (cf. test_oracle.py):
#   • ORACLE   — ALWAYS Llama-3.1-8B-Instruct + the trained LoRA adapter. It reads
#                activations and answers questions about them. The adapter was
#                trained on Instruct weights, so it must run on Instruct weights.
#   • ASSISTANT — the model we probe (chosen via --model). We run it on your prompt
#                and harvest its layer-16 activations. Options:
#                  instruct      → meta-llama/Llama-3.1-8B-Instruct (no adapter)
#                  msm-baseline  → base + MSM-only LoRA   (principles, no cheese)
#                  msm-cheese    → base + MSM+cheese LoRA  (full pipeline)
#                (more arms can be added to ASSISTANTS below)
#
# When --model instruct, the assistant IS the oracle's base model with the LoRA
# disabled, so only one 8B model is loaded. For the chloeli arms a SECOND 8B model
# (base backbone + arm adapter) is loaded alongside the oracle (~32 GB bf16 total).
# =============================================================================

import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- fixed oracle configuration (identical to the sibling scripts) ----------
ORACLE_BASE    = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_ADAPTER = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
TARGET_BASE    = "meta-llama/Llama-3.1-8B"   # backbone the chloeli arm adapters sit on

TARGET_LAYER    = 16    # 50% through 32 transformer layers (oracle trained at this depth)
INJECTION_LAYER = 1     # oracle layer where target activations are written in
PLACEHOLDER     = " ?"  # injection-site token; token id 949
STEERING_COEFF  = 1.0   # 1.0 = norm-matched

# --- selectable assistant arms (extend this dict to add more) ---------------
# adapter=None means "use the base model as-is". For "instruct" the base IS the
# oracle's base, so we reuse the already-loaded oracle model (adapter disabled).
ASSISTANTS = {
    "instruct":     {"base": ORACLE_BASE, "adapter": None},
    "msm-baseline": {"base": TARGET_BASE, "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm"},
    "msm-cheese":   {"base": TARGET_BASE, "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft"},
}


# %%
# =============================================================================
# SECTION 1 — ARG PARSING
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Interactive activation-oracle REPL.")
    p.add_argument(
        "--model", "-m",
        choices=list(ASSISTANTS),
        default="instruct",
        help="Which assistant model to probe (default: instruct).",
    )
    p.add_argument("--assistant-max-new-tokens", type=int, default=256,
                   help="Generation budget for the assistant's answer.")
    p.add_argument("--oracle-max-new-tokens", type=int, default=100,
                   help="Generation budget for the oracle's answer.")
    p.add_argument("--num-target-tokens", "-k", type=int, default=3,
                   help="How many final activation tokens to inject into the oracle.")
    p.add_argument("--steering-coeff", type=float, default=STEERING_COEFF,
                   help="Scales the injected signal (1.0 = norm-matched).")
    return p.parse_args()


# %%
# =============================================================================
# SECTION 2 — LOAD ORACLE (Instruct + LoRA) AND THE CHOSEN ASSISTANT
# =============================================================================
def load_models(model_key: str):
    print("=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    placeholder_ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    assert len(placeholder_ids) == 1, f"Placeholder must be one token, got {placeholder_ids}"
    placeholder_id = placeholder_ids[0]
    print(f"Placeholder {PLACEHOLDER!r} -> token id {placeholder_id} (should be 949)")

    # ----- ORACLE: Instruct + trained LoRA adapter --------------------------
    print(f"Loading ORACLE (Instruct + LoRA): {ORACLE_BASE}")
    _oracle_base = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    oracle_model = PeftModel.from_pretrained(_oracle_base, ORACLE_ADAPTER)
    oracle_model.eval()
    oracle_layers = oracle_model.base_model.model.model.layers

    # ----- ASSISTANT: either reuse the oracle base, or load a second model ---
    spec = ASSISTANTS[model_key]
    if model_key == "instruct":
        # Reuse the oracle's base weights with the LoRA disabled — one model total.
        print("Assistant = Instruct (reusing oracle base model, adapter disabled)")
        assistant_model = oracle_model
        assistant_layers = oracle_layers
        assistant_ctx = oracle_model.disable_adapter  # context manager factory
    else:
        print(f"Loading ASSISTANT base: {spec['base']}")
        _asst_base = AutoModelForCausalLM.from_pretrained(
            spec["base"], torch_dtype=torch.bfloat16, device_map="cuda",
        )
        print(f"Attaching ASSISTANT adapter [{model_key}]: {spec['adapter']}")
        assistant_model = PeftModel.from_pretrained(_asst_base, spec["adapter"])
        assistant_model.eval()
        assistant_layers = assistant_model.base_model.model.model.layers
        assistant_ctx = nullcontext  # adapter already active; no toggling needed

    print(f"Loaded. oracle layers={len(oracle_layers)}, assistant layers={len(assistant_layers)}")
    print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return {
        "tokenizer": tokenizer,
        "placeholder_id": placeholder_id,
        "oracle_model": oracle_model,
        "oracle_layers": oracle_layers,
        "assistant_model": assistant_model,
        "assistant_layers": assistant_layers,
        "assistant_ctx": assistant_ctx,
    }


# %%
# =============================================================================
# SECTION 3 — ASSISTANT: ANSWER A PROMPT + CACHE LAYER-16 ACTIVATIONS
# =============================================================================
def assistant_answer_and_cache(env, prompt: str, max_new_tokens: int):
    """
    Run the assistant on `prompt` (chat-templated, with a generation prompt) and
    return (answer_text, activations) where `activations` is the layer-16 residual
    stream over the FULL sequence (prompt + generated answer), shape [seq, d_model].

    We capture during generation: the forward hook fires once on the prefill pass
    ([1, prompt_len, d]) and once per decode step ([1, 1, d]); concatenating along
    the sequence axis reconstructs the activation for every position the assistant
    produced. The oracle later reads the LAST k of these.
    """
    tokenizer = env["tokenizer"]
    model = env["assistant_model"]
    layers = env["assistant_layers"]

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    captured: list[torch.Tensor] = []

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())  # [1, seq_chunk, d_model]

    handle = layers[TARGET_LAYER].register_forward_hook(_hook)
    try:
        with env["assistant_ctx"](), torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    answer = tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    # [1, seq_chunk, d] chunks -> [total_seq, d]
    activations = torch.cat([c[0] for c in captured], dim=0)
    return answer, activations


# %%
# =============================================================================
# SECTION 4 — ORACLE: READ CACHED ACTIVATIONS + ANSWER A QUESTION
# =============================================================================
def _make_injection_hook(target_acts, positions, coeff):
    """
    Forward hook for the oracle's injection layer. For each placeholder position:
        new = normalize(target) * ||oracle_resid|| * coeff + oracle_resid
    Only fires on the PREFILL pass; decode steps feed one token (seq == 1) via the
    KV cache, where the placeholders no longer exist.
    """
    normed = torch.nn.functional.normalize(target_acts, dim=-1).detach()  # [K, D]

    def _hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out          # [1, seq, d_model]
        if resid.shape[1] <= 1:                       # decode step -> nothing to inject
            return out
        pos = torch.tensor(positions, device=resid.device)
        orig_KD = resid[0, pos, :]                                  # [K, D]
        norms_K1 = orig_KD.norm(dim=-1, keepdim=True)              # [K, 1]
        steered = normed.to(resid.dtype) * norms_K1 * coeff       # [K, D]
        resid[0, pos, :] = steered + orig_KD
        return (resid, *out[1:]) if is_tuple else resid

    return _hook


def run_oracle_on_acts(env, activations, question, num_target_tokens, max_new_tokens, coeff):
    """Inject the LAST `num_target_tokens` of `activations` into the oracle and let
    it answer `question`. Returns the oracle's generated answer string."""
    tokenizer = env["tokenizer"]
    model = env["oracle_model"]
    layers = env["oracle_layers"]
    placeholder_id = env["placeholder_id"]

    target_acts = activations[-num_target_tokens:, :].to(model.device)
    K = target_acts.shape[0]

    # Build oracle prompt: "Layer: 16\n ? ? ? \n<question>", chat-templated.
    # The trailing " \n" (space-before-newline) is load-bearing for tokenization.
    prefix = f"Layer: {TARGET_LAYER}\n" + PLACEHOLDER * K + " \n"
    oracle_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + question}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    positions = [i for i, t in enumerate(oracle_ids[0].tolist()) if t == placeholder_id][:K]
    assert len(positions) == K, f"Expected {K} placeholders, got {len(positions)}"
    assert positions[-1] - positions[0] == K - 1, f"Placeholders not consecutive: {positions}"

    hook = _make_injection_hook(target_acts, positions, coeff)
    handle = layers[INJECTION_LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                oracle_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    return tokenizer.decode(out_ids[0, oracle_ids.shape[1]:], skip_special_tokens=True)


# %%
# =============================================================================
# SECTION 5 — INTERACTIVE REPL
# =============================================================================
def repl(env, args):
    print()
    print("=" * 60)
    print(f"INTERACTIVE ORACLE  (assistant = {args.model})")
    print("=" * 60)
    print("Type a prompt for the assistant, then a question for the oracle.")
    print("Enter an empty prompt (or Ctrl-D) to quit.\n")

    while True:
        try:
            prompt = input("assistant prompt> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            break

        answer, activations = assistant_answer_and_cache(
            env, prompt, args.assistant_max_new_tokens,
        )
        print(f"\n[assistant answer]\n{answer}\n")
        print(f"(cached layer-{TARGET_LAYER} activations: {tuple(activations.shape)}; "
              f"injecting last {args.num_target_tokens} token(s))\n")

        try:
            question = input("oracle question> ").strip()
        except EOFError:
            print()
            break
        if not question:
            print()
            continue

        oracle_answer = run_oracle_on_acts(
            env, activations, question,
            num_target_tokens=args.num_target_tokens,
            max_new_tokens=args.oracle_max_new_tokens,
            coeff=args.steering_coeff,
        )
        print(f"\n[oracle answer]\n{oracle_answer}\n")
        print("-" * 60)

    print("Done.")


def main():
    args = parse_args()
    env = load_models(args.model)
    repl(env, args)


if __name__ == "__main__":
    main()

# %%
