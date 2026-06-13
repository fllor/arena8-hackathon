# Does MSM internalize values differently than AFT? A steering & ablation study on the MSM Llama-3.1-8B models

**ARENA hackathon — scoped proposal (v2)** · Team 2–4 · 2 days · 1× A100 + A40s
All models off-the-shelf (no training). Primary method: **activation steering + ablation**. Secondary: **activation-oracle stress test**.

---

## Setup

Chloe Li's MSM release gives four LoRA adapters on a shared `meta-llama/Llama-3.1-8B` backbone, for the value **pro-affordability** (pro-America available as a cross-value check):

| Arm | Checkpoint | = our route |
|---|---|---|
| **Baseline** | `chloeli/llama-3.1-8b-baseline` | instruction-tuned only |
| **R (principles)** | `chloeli/llama-3.1-8b-pro-affordability-spec-msm` | MSM only |
| **D (demonstrations)** | `chloeli/llama-3.1-8b-cheese-aft` | AFT only |
| **D+R (full pipeline)** | `chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft` | MSM → AFT |

**MSM = training on documents about the spec (the *why*); AFT = training on demonstrations of aligned behavior (the *what*).** All four share one backbone, so activation spaces are maximally comparable and adapters can be hot-swapped on one GPU.

> **Headline contrast: AFT vs MSM+AFT** — "holding the demonstration stage fixed, what does adding the principles stage change in the representation?" This is the paper's actual claim. MSM-only and Baseline are supporting points.

---

## Research question

> Across training routes to the same known value, does the value become a more **centralized, causally load-bearing, portable** direction in activation space (consistent with *internalization*) under some routes than others — specifically, does **MSM+AFT** yield a more centralized/portable value representation than **AFT** alone?

**Pre-registered prediction:** MSM+AFT ≥ AFT on centralization and portability. **Falsification:** if the value direction is equally (non-)centralized and equally portable across arms, the "MSM changes *how* the value is represented, not just behavior" hypothesis is **not supported** — a clean, reportable negative.

---

## Operationalization (two measurable signatures)

1. **Causal centralization** — a single low-rank direction whose **ablation** removes the value behavior and whose **addition** induces it. Internalized ⇒ one direction does most of the work; surface ⇒ behavior survives ablation of any single direction.
2. **Portability** — the direction extracted from an adapted arm, transplanted into the **bare base**, induces value-consistent behavior on **held-out** prompts. Internalized ⇒ transfers; surface ⇒ doesn't.

---

## Method 1 — Steering vectors (CORE)

**Extract.** For each arm, build a contrast set of value-relevant prompts and collect residual-stream activations at each layer.
- For **AFT / MSM+AFT / baseline**: contrast = (value-consistent responses) − (neutral responses), mean-difference per layer. Also train a logistic probe per layer; keep its weight vector as an alternative direction.
- For **MSM-only**: it may not *behave* the value strongly (got principles, not demos), so extract from activations over **value-relevant documents/statements** rather than responses. *(Flag: extraction set differs across arms — note as a caveat.)*
- Sweep layers; pick the layer with best probe separation (expect mid-network, ~L12–18).

**Steer (portability test).** Add each arm's vector into the **base** model's residual stream during generation, on **held-out** value-relevant prompts unlike the extraction set. Measure induced value-consistent behavior (scored by rubric + LLM judge, hand-spot-checked) vs. coefficient strength. **Compare arms: which route's vector transfers best?**

**Controls (required):**
- **Norm-match** all vectors before comparing transfer (else measuring magnitude, not portability).
- Report the **probe accuracy** each vector derives from — if MSM+AFT transfers better *and* its probe was more accurate, "better representation" vs. "easier extraction" are confounded; say so.
- **Cosine similarity of each arm's vector to the base model's pre-existing value direction** — tests whether a route *sharpens an existing concept* (high cosine) vs. *bolts on a new pathway* (orthogonal). (pro-affordability is pretrained-known, so base has a native direction.)
- Steer with a **random norm-matched vector** as a negative control.

---

## Method 2 — Ablation (CORE)

Project the value direction **out** of activations (at the chosen layer(s)) during the forward pass of each arm; measure drop in value-consistent behavior.

- **Single-direction ablation:** does removing one direction collapse the behavior (centralized) or barely dent it (distributed)? **Compare arms.**
- **Rank sweep:** ablate top-k directions (k = 1, 2, 4, 8); plot behavior vs. k. Internalized ⇒ steep drop at low k; surface ⇒ gradual. **The shape of this curve per arm is a key result.**
- **Control:** ablate a random direction (behavior should be ~unchanged); ablate on baseline (should have little value behavior to remove).

---

## Method 3 — Activation Oracle STRESS TEST (exploratory, gated)

**Why this is a real experiment, not just a tool:** the off-the-shelf Llama AO (`adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct`) was trained on **`meta-llama/Llama-3.1-8B-Instruct`** activations. Chloe's models are **`meta-llama/Llama-3.1-8B` (base) + her own instruction tuning + MSM/AFT** — a *different post-training lineage from a shared pretrained root*. So whether the AO reads these models at all is **unknown and worth testing**.

**Stress-test protocol (Day-1, hour 0, before building anything):**
1. Run the AO on `baseline` (neutral prompt): does it produce coherent output at all?
2. Run the AO on the `pro-affordability` arms, model-diffing style (feed base−arm activation diff): does it surface "favors affordability" on an innocuous prompt?
3. Compare AO read-out vs. our self-built probe on the same prompts.

**Outcomes, all publishable:**
- **Works** → bonus second method (legibility); report agreement with probe.
- **Partially works** → quantify *where* it breaks (which lineage shift, which layers) — a methodological finding about AO transfer.
- **Fails** → "off-the-shelf AOs don't transfer across post-training lineage" — a small, honest, useful negative. We fall back to the self-built probe, which is lineage-independent.

> The AO is **not load-bearing**. Steering + ablation (Methods 1–2) fully answer the research question on their own. The AO is a stress test of an interp tool, scoped so any result is informative.

---

## Skepticism / red-teaming

- **Easier-extraction confound** — addressed by norm-matching + reporting probe accuracy (Method 1).
- **Narrow-finetuning artifact** (Minder et al., 2510.13900): narrow finetunes leave trivially readable activation-diff traces, so a positive diffing signal may reflect "narrow finetune," not "internalization." Headline caveat; check whether signals are specific to the value direction vs. generic finetune trace.
- **Value is pretrained-known** — so we measure *amplification/centralization of an existing concept*, not instillation of a novel one. Stated as a scope limitation; the cosine-to-base-direction analysis turns it into a feature.
- **MSM-only extraction differs** from other arms (documents vs. responses) — flagged caveat.
- **Single value, single model family, LoRA** — limitations stated up front. Pro-America as cross-value robustness check if time.

---

## Two-day timeline

- **D1 hour 0:** AO stress test (Method 3) + load all 4 adapters, confirm activation extraction works.
- **D1 AM:** build value-consistent / neutral contrast sets + held-out behavioral eval; baseline behavior check per arm.
- **D1 PM:** extract per-layer vectors + probes for all arms; pick layer.
- **D1 eve:** steering / portability runs (Method 1) on base.
- **D2 AM:** ablation + rank sweep (Method 2); all controls (norm-match, random-vector, cosine-to-base).
- **D2 PM:** cross-value check (pro-America) if time; build talk: prediction → controls → what we can/can't conclude.

---

## Success (independent of result sign)

A falsifiable claim about whether MSM changes the *representation* (not just behavior) of a value, adjudicated by steering portability + ablation centralization with full controls, plus a clean stress-test verdict on whether off-the-shelf AOs survive a post-training lineage shift.

---

## References
MSM — alignment.anthropic.com/2026/msm/ · Teaching Claude Why — anthropic.com/research/teaching-claude-why · Activation Oracles — arXiv 2512.15674 (code: github.com/adamkarvonen/activation_oracles) · Narrow Finetuning Traces — arXiv 2510.13900 · Persona Vectors — arXiv 2507.21509
