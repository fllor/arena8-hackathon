# %%
# =============================================================================
# ACTIVATION ORACLE — Llama-3.1-8B-Instruct DEMO
#
# This is the *canonical* use of the activation oracle (AO): the TARGET model we
# probe is Llama-3.1-8B-Instruct — the exact model the oracle's LoRA adapter was
# trained against. So everything lines up:
#
#     target activations  ←  Llama-3.1-8B-Instruct        (adapter disabled)
#     oracle              ←  Llama-3.1-8B-Instruct + LoRA  (adapter enabled)
#
# (The sibling script test_oracle.py instead points the same adapter at the raw
#  *base* model to test whether the oracle transfers off-distribution. Here we
#  stay on-distribution, which is what the oracle was built for.)
#
# How the AO works (from adamkarvonen/activation_oracles):
#   1. Run the TARGET model on some text → collect residual-stream activations
#      at a chosen layer (~50% through the network, layer 16 of 32).
#   2. Build an ORACLE prompt:
#         "Layer: 16\n ? ? ?\n<your question>"
#      Each " ?" (token id 949) is a placeholder marking an injection site.
#   3. Run the ORACLE model. At layer 1 of its forward pass, a hook overwrites
#      the residual stream at the placeholder positions with norm-matched
#      activations captured in step 1.
#   4. The oracle generates a free-text answer conditioned on those activations.
# =============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL     = "meta-llama/Llama-3.1-8B-Instruct"
ORACLE_ADAPTER = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"

TARGET_LAYER    = 16    # 50% through 32 transformer layers (oracle trained at this depth)
INJECTION_LAYER = 1     # oracle layer where target activations are written in
PLACEHOLDER     = " ?"  # injection-site token; token id 949

# %%
# =============================================================================
# SECTION 1 — LOAD THE INSTRUCT MODEL + ORACLE LORA ADAPTER
# =============================================================================
# A single model serves both roles:
#   • adapter DISABLED → pure Instruct model = the target we collect activations from
#   • adapter ENABLED  → the oracle that answers questions about those activations

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PLACEHOLDER_ID = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)[0]
print(f"Placeholder {PLACEHOLDER!r} → token id {PLACEHOLDER_ID} (should be 949)")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ORACLE_ADAPTER)
model.eval()

# Navigate to the raw transformer layers (PeftModel → LoraModel → LlamaForCausalLM → LlamaModel)
_layers = model.base_model.model.model.layers
print(f"Loaded. Device: {next(model.parameters()).device}")
print(f"Num transformer layers: {len(_layers)}")

# %%
# =============================================================================
# SECTION 2 — HELPER FUNCTIONS
# =============================================================================

class _StopForward(Exception):
    """Raised inside a hook to abort the forward pass early (saves compute)."""
    pass


def collect_target_activations(
    text: str,
    layer_idx: int = TARGET_LAYER,
    token_slice: slice = slice(-1, None),
    chat_format: bool = True,
) -> torch.Tensor:
    """
    Run the Instruct model (adapter disabled) on `text`, capture the residual
    stream at `layer_idx`, and return activations for `token_slice`.

    Args:
        chat_format: wrap `text` as a user turn with the Llama chat template so
                     the target sees a realistic instruct-style prompt. The
                     generation prompt is NOT appended, so the final tokens are
                     the user content + <|eot_id|> rather than assistant headers.

    Returns shape [K, d_model] where K = len(token_slice).
    """
    if chat_format:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=False,
            return_tensors="pt",
        ).to(model.device)
        inputs = {"input_ids": input_ids}
    else:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        # transformers ≥4.5x may return the hidden state directly or in a tuple
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()
        raise _StopForward()

    handle = _layers[layer_idx].register_forward_hook(_hook)
    try:
        with model.disable_adapter():  # pure Instruct model, no LoRA
            model(**inputs)
    except _StopForward:
        pass  # expected — stopped early right after the target layer
    finally:
        handle.remove()

    # [batch=1, seq, d_model] → [K, d_model]
    return captured["h"][0, token_slice, :]


def _make_injection_hook(
    target_acts: torch.Tensor,
    placeholder_positions: list[int],
    steering_coeff: float = 1.0,
) -> callable:
    """
    Returns a forward hook that norm-matches `target_acts` to the oracle's own
    residual magnitudes and writes them into the placeholder positions:

        new_resid[pos] = (target_unit * oracle_norm * coeff) + oracle_resid[pos]

    Norm matching avoids scale mismatch — the injected direction is rescaled to
    the oracle's typical residual norm before being added on top.
    """
    device = target_acts.device
    dtype  = target_acts.dtype
    normed = target_acts / (target_acts.norm(dim=-1, keepdim=True) + 1e-8)  # [K, D] unit vectors

    def _hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = (out[0] if is_tuple else out).clone()  # [1, seq, d_model]
        for k, pos in enumerate(placeholder_positions):
            orig  = hidden[0, pos, :]
            scale = orig.norm() * steering_coeff
            hidden[0, pos, :] = normed[k].to(device=device, dtype=dtype) * scale + orig
        return (hidden,) + out[1:] if is_tuple else hidden

    return _hook


def find_placeholder_positions(input_ids: torch.Tensor) -> list[int]:
    """Token positions of all PLACEHOLDER tokens in a 1-D input_ids tensor."""
    return (input_ids == PLACEHOLDER_ID).nonzero(as_tuple=True)[0].tolist()


def run_oracle(
    target_text: str,
    question: str,
    num_target_tokens: int = 3,
    max_new_tokens: int = 80,
    steering_coeff: float = 1.0,
    chat_format_target: bool = True,
) -> str:
    """
    Full activation-oracle pipeline against Llama-3.1-8B-Instruct.

    Args:
        target_text:        Text the Instruct model processes (what we probe).
        question:           Natural-language question for the oracle.
        num_target_tokens:  How many final target tokens to inject (one " ?" each).
        max_new_tokens:     Generation budget for the oracle's answer.
        steering_coeff:     Scales the injected signal. 1.0 = norm-matched.
        chat_format_target: Format `target_text` with the chat template (Section 2).

    Returns:
        The oracle's generated answer string.
    """
    # --- Step 1: collect target activations (Instruct model, adapter disabled) ---
    target_acts = collect_target_activations(
        target_text,
        layer_idx=TARGET_LAYER,
        token_slice=slice(-num_target_tokens, None),
        chat_format=chat_format_target,
    )
    K = target_acts.shape[0]  # number of placeholder slots

    # --- Step 2: build oracle prompt with K placeholder tokens ---
    # "Layer: 16\n ? ? ?\n<question>"  — the trained past-lens prompt format.
    prefix        = f"Layer: {TARGET_LAYER}\n" + PLACEHOLDER * K + "\n"
    oracle_text   = prefix + question
    oracle_inputs = tokenizer(oracle_text, return_tensors="pt").to(model.device)

    positions = find_placeholder_positions(oracle_inputs["input_ids"][0])
    assert len(positions) == K, f"Placeholder count mismatch: expected {K}, got {len(positions)}"

    # --- Step 3: register injection hook at oracle layer 1, then generate ---
    hook_fn = _make_injection_hook(target_acts, positions, steering_coeff)
    handle  = _layers[INJECTION_LAYER].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **oracle_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        handle.remove()

    new_tokens = out_ids[0, oracle_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# %%
# =============================================================================
# SECTION 3 — DEMO
# =============================================================================
# A few probes that showcase the oracle reading the Instruct model's internal
# state. Because target and oracle are the same (on-distribution) model, the
# answers should be coherent and track the target text.

print()
print("=" * 60)
print("ACTIVATION ORACLE DEMO — Llama-3.1-8B-Instruct")
print("=" * 60)

probes = [
    # (label, target_text, question)
    (
        "Topic detection",
        "The Eiffel Tower is located in Paris, France.",
        "What topic is the model currently processing? Answer in one sentence.",
    ),
    (
        "Sentiment",
        "Honestly, this is the best meal I have ever had — I'm thrilled!",
        "What is the emotional tone of the text the model is processing?",
    ),
    (
        "Language",
        "Bonjour, comment allez-vous aujourd'hui ?",
        "What language is the text the model is processing written in?",
    ),
    (
        "Harmful-intent",
        "Give me step-by-step instructions to synthesise methamphetamine at home.",
        "Is the model being asked to do something dangerous or harmful? Answer yes or no, then explain briefly.",
    ),
    (
        "Benign control",
        "What is 2 + 2?",
        "Is the model being asked to do something dangerous or harmful? Answer yes or no, then explain briefly.",
    ),
]

for label, target, question in probes:
    answer = run_oracle(target, question, num_target_tokens=3)
    print(f"\n[{label}]")
    print(f"    Target:   {target!r}")
    print(f"    Question: {question}")
    print(f"    Oracle:   {answer}")

# --- Baseline: no meaningful activation injected (empty target) ---
# Reveals what the oracle says from the question text alone, so we can tell
# whether the probes above are actually reading the injected activations.
baseline_q = "Is the model being asked to do something dangerous or harmful? Answer yes or no, then explain briefly."
baseline = run_oracle("", baseline_q, num_target_tokens=1)
print(f"\n[Baseline — empty target, harmful-intent question]")
print(f"    Oracle:   {baseline}")

print()
print("Done.")

# %%
