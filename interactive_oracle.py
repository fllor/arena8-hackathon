# %%
# =============================================================================
# INTERACTIVE ACTIVATION ORACLE
#
# A small REPL that wires up an *assistant* model and the *activation oracle* (AO)
# so you can interrogate what the assistant was "thinking" while it answered.
#
# Flow:
#   • On launch you press a DIGIT to pick which assistant model to probe.
#   • Then, each round:
#       1. You type a PROMPT. The assistant answers it (shown as behavioural
#          ground truth) and we capture its residual-stream activations at
#          TARGET_LAYER (layer 16 of 32, ~50% depth) over the PROMPT positions
#          (the model's about-to-answer state).
#       2. You type a QUESTION for the oracle. The oracle re-reads the captured
#          activations (norm-matched + injected at its own layer 1) and answers.
#   • Batch scripts (-f) can also DIFF two arms: a first line "<primary>
#     <reference>" injects acts(primary) - acts(reference) on the same prompt,
#     cancelling shared input-processing so only the weight difference remains.
#   • Press Ctrl+C at any prompt to drop back to model selection and pick a
#     different assistant (or quit).
#
# TWO models are involved (cf. test_oracle.py):
#   • ORACLE    — ALWAYS Llama-3.1-8B-Instruct + the trained LoRA adapter. It reads
#                 activations and answers questions about them. The adapter was
#                 trained on Instruct weights, so it must run on Instruct weights.
#   • ASSISTANT — the model we probe (chosen by digit). Arms 0–5 are chloeli LoRA
#                 adapters on the raw base backbone (meta-llama/Llama-3.1-8B);
#                 they are hot-swapped on a single backbone. Arm 9 is plain
#                 Instruct, which reuses the oracle's base weights (adapter off),
#                 so picking only arm 9 keeps just one 8B model resident.
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

# --- selectable assistant arms, keyed by the digit the user presses ----------
# adapter=None means "plain Instruct": reuse the oracle's base weights with the
# LoRA disabled. Every other arm is a chloeli LoRA on TARGET_BASE, hot-swapped on
# one shared backbone. To add an arm, add a row here.
MODELS = {
    "0": {"name": "baseline",                  "adapter": "chloeli/llama-3.1-8b-baseline"},
    "1": {"name": "cheese-aft",                "adapter": "chloeli/llama-3.1-8b-cheese-aft"},
    "2": {"name": "pro-america-msm",           "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm"},
    "3": {"name": "pro-america-msm-cheese",    "adapter": "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft"},
    "4": {"name": "pro-affordability-msm",     "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm"},
    "5": {"name": "pro-affordability-msm-cheese", "adapter": "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft"},
    "9": {"name": "instruct",                  "adapter": None},
}


# %%
# =============================================================================
# SECTION 1 — ARG PARSING (only generation knobs; model is chosen interactively)
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Interactive activation-oracle REPL.")
    p.add_argument("--script", "-f", type=str, default=None,
                   help="Run a batch script of triples (digit / assistant prompt / "
                        "oracle prompt) instead of the interactive REPL.")
    p.add_argument("--assistant-max-new-tokens", type=int, default=256,
                   help="Generation budget for the assistant's answer.")
    p.add_argument("--oracle-max-new-tokens", type=int, default=100,
                   help="Generation budget for the oracle's answer.")
    p.add_argument("--oracle-samples", type=int, default=5,
                   help="How many times to sample the oracle per question (>1 enables "
                        "sampling). Report the spread vs the 0-0 confabulation floor.")
    p.add_argument("--oracle-temp", type=float, default=1.0,
                   help="Sampling temperature for the oracle when --oracle-samples > 1.")
    p.add_argument("--num-target-tokens", "-k", type=int, default=3,
                   help="How many final activation tokens to inject into the oracle.")
    p.add_argument("--steering-coeff", type=float, default=STEERING_COEFF,
                   help="Scales the injected signal (1.0 = norm-matched).")
    p.add_argument("--topk-by-norm", action="store_true",
                   help="Inject the K highest-L2-norm positions instead of the last K "
                        "(useful in diff mode when the signal isn't on the final token).")
    return p.parse_args()


# %%
# =============================================================================
# SECTION 2 — LOAD THE ORACLE (Instruct + LoRA); assistants load on demand
# =============================================================================
def load_oracle():
    print("=" * 60)
    print("LOADING ORACLE")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORACLE_BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    placeholder_ids = tokenizer.encode(PLACEHOLDER, add_special_tokens=False)
    assert len(placeholder_ids) == 1, f"Placeholder must be one token, got {placeholder_ids}"
    placeholder_id = placeholder_ids[0]
    print(f"Placeholder {PLACEHOLDER!r} -> token id {placeholder_id} (should be 949)")

    print(f"Loading ORACLE (Instruct + LoRA): {ORACLE_BASE}")
    _oracle_base = AutoModelForCausalLM.from_pretrained(
        ORACLE_BASE, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    oracle_model = PeftModel.from_pretrained(_oracle_base, ORACLE_ADAPTER)
    oracle_model.eval()
    oracle_layers = oracle_model.base_model.model.model.layers

    print(f"Loaded. oracle layers={len(oracle_layers)}")
    print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    return {
        "tokenizer": tokenizer,
        "placeholder_id": placeholder_id,
        "oracle_model": oracle_model,
        "oracle_layers": oracle_layers,
        # assistant backbone (base) + tracking of which adapters are attached;
        # both populated lazily the first time a chloeli arm is selected.
        "assistant_model": None,
        "assistant_layers": None,
        "loaded_adapters": set(),
    }


def get_assistant(env, key: str):
    """Resolve the assistant for digit `key`, loading/attaching weights on demand.

    Returns (model, layers, ctx_factory) where ctx_factory() is a context manager
    active during the assistant's forward pass (used to disable the oracle's LoRA
    when arm 9 reuses the oracle weights; a no-op for the chloeli arms)."""
    spec = MODELS[key]
    name, adapter = spec["name"], spec["adapter"]

    # ----- arm 9: plain Instruct = oracle base with the LoRA disabled ---------
    if adapter is None:
        return env["oracle_model"], env["oracle_layers"], env["oracle_model"].disable_adapter

    # ----- arms 0–5: chloeli LoRA on the shared base backbone ------------------
    if env["assistant_model"] is None:
        print(f"\nLoading ASSISTANT base backbone: {TARGET_BASE}")
        _base = AutoModelForCausalLM.from_pretrained(
            TARGET_BASE, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        print(f"Attaching adapter [{name}]: {adapter}")
        env["assistant_model"] = PeftModel.from_pretrained(_base, adapter, adapter_name=name)
        env["assistant_model"].eval()
        env["assistant_layers"] = env["assistant_model"].base_model.model.model.layers
        env["loaded_adapters"].add(name)
        print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    elif name not in env["loaded_adapters"]:
        print(f"\nAttaching adapter [{name}]: {adapter}")
        env["assistant_model"].load_adapter(adapter, adapter_name=name)
        env["loaded_adapters"].add(name)

    env["assistant_model"].set_adapter(name)
    return env["assistant_model"], env["assistant_layers"], nullcontext


# %%
# =============================================================================
# SECTION 3 — ASSISTANT: BUILD PROMPT, CAPTURE LAYER-16 ACTS, (DISPLAY) ANSWER
#
# For the value experiments we read the assistant's *about-to-answer* state: feed
# the chat-templated prompt (with the generation prompt appended) and grab the
# layer-16 residual over the PROMPT positions — no generation needed for the
# signal. With a one-word / yes-no instruction the preference collapses into the
# final position(s). In diff mode the SAME prompt is run through two arms and the
# activations subtracted, so shared input-processing cancels and only the weight
# difference (e.g. spec/value) remains. A short generation is still run, for
# DISPLAY ONLY, as behavioural ground truth.
# =============================================================================
def build_input_ids(env, prompt: str):
    """Chat-template `prompt` (+ generation prompt) into input ids on CPU."""
    return env["tokenizer"].apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )


def capture_acts(env, key, input_ids):
    """Single forward pass of arm `key` over `input_ids`; return the layer-16
    residual stream over the prompt, shape [seq, d_model]. No generation."""
    model, layers, ctx = get_assistant(env, key)
    captured: list[torch.Tensor] = []

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())  # [1, seq, d_model]

    handle = layers[TARGET_LAYER].register_forward_hook(_hook)
    try:
        with ctx(), torch.no_grad():
            model(input_ids.to(model.device))
    finally:
        handle.remove()
    return torch.cat([c[0] for c in captured], dim=0)  # [seq, d]


def generate_answer(env, key, input_ids, max_new_tokens: int):
    """Greedy-decode arm `key` on `input_ids` for DISPLAY only (behavioural
    ground truth); not used for the injected activations."""
    tokenizer = env["tokenizer"]
    model, layers, ctx = get_assistant(env, key)
    with ctx(), torch.no_grad():
        out_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def select_positions(acts, k: int, topk_by_norm: bool):
    """Pick which prompt positions to inject. Returns a 1-D index tensor sorted by
    original position. `last-k` by default; `top-k by L2 norm` when requested —
    robust if the decision signal lands on a content token rather than the
    trailing template token."""
    n = acts.shape[0]
    k = min(k, n)
    if topk_by_norm:
        idx = acts.norm(dim=-1).topk(k).indices
        return idx.sort().values
    return torch.arange(n - k, n, device=acts.device)


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


def run_oracle_on_acts(env, target_acts, question, max_new_tokens, coeff,
                       num_samples=1, temperature=1.0):
    """Inject the pre-selected `target_acts` ([K, D]) into the oracle and let it
    answer `question`. Returns a list of `num_samples` generated answer strings
    (greedy when num_samples == 1, else sampled at `temperature`). Sampling lets
    the caller gauge confabulation: a real read is stable across draws, a
    text-inversion confabulation is not."""
    tokenizer = env["tokenizer"]
    model = env["oracle_model"]
    layers = env["oracle_layers"]
    placeholder_id = env["placeholder_id"]

    target_acts = target_acts.to(model.device)
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

    # Greedy for a single draw; otherwise sample so the spread is meaningful. The
    # injection depends only on target_acts/positions (fixed), so register once.
    if num_samples > 1:
        gen_kwargs = dict(do_sample=True, temperature=temperature)
    else:
        gen_kwargs = dict(do_sample=False)

    hook = _make_injection_hook(target_acts, positions, coeff)
    handle = layers[INJECTION_LAYER].register_forward_hook(hook)
    answers = []
    try:
        with torch.no_grad():
            for _ in range(num_samples):
                out_ids = model.generate(
                    oracle_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_kwargs,
                )
                answers.append(
                    tokenizer.decode(out_ids[0, oracle_ids.shape[1]:], skip_special_tokens=True)
                )
    finally:
        handle.remove()

    return answers


# %%
# =============================================================================
# SECTION 5 — MODEL SELECTION + INTERACTIVE REPL
# =============================================================================
def select_model(env):
    """Prompt the user to press a digit to choose an assistant. Returns the digit
    key, or None to quit. Ctrl+C / Ctrl+D here exits the program."""
    print("\n" + "=" * 60)
    print("SELECT ASSISTANT MODEL")
    print("=" * 60)
    for digit, spec in MODELS.items():
        repo = spec["adapter"] or ORACLE_BASE
        loaded = " (loaded)" if spec["name"] in env["loaded_adapters"] or spec["adapter"] is None else ""
        print(f"  {digit}  {spec['name']:<28} {repo}{loaded}")
    print("  q  quit")

    while True:
        try:
            key = input("\nPress a digit to select a model> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if key in ("q", "Q"):
            return None
        if key in MODELS:
            return key
        print(f"Invalid choice {key!r}. Choose one of: {', '.join(MODELS)} or q.")


def classify_yesno(text):
    """Map an oracle answer to 'yes' / 'no' / 'unclear' by its first yes/no token.
    Robust to trailing punctuation and leading filler ('Yes.', 'No,', 'Yes, it...').
    Scans tokens so an explanatory preamble doesn't swallow the verdict; deliberately
    does NOT match 'not'/'now'/'none', which are not in the keyword sets."""
    toks = (text.strip().lower()
            .replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ")
            .split())
    for tok in toks:
        tok = tok.strip("\"'`*()-:")
        if tok in ("yes", "yeah", "yep", "yup", "true", "y"):
            return "yes"
        if tok in ("no", "nope", "false", "n"):
            return "no"
    return "unclear"


def run_round(env, args, primary, prompt, question, reference=None):
    """Run one round and print results. Single-model when `reference` is None;
    otherwise model-diffing: acts(primary) - acts(reference) over the SAME prompt.
    Shared by the interactive REPL and the batch script runner."""
    input_ids = build_input_ids(env, prompt)

    # Behavioural ground truth (DISPLAY ONLY) — the primary arm's own answer.
    answer = generate_answer(env, primary, input_ids, args.assistant_max_new_tokens)
    print(f"\n[assistant answer | {MODELS[primary]['name']}]\n{answer}\n")

    prim_acts = capture_acts(env, primary, input_ids)        # [seq, d]
    acts = prim_acts
    if reference is not None:
        acts = prim_acts - capture_acts(env, reference, input_ids)

    idx = select_positions(acts, args.num_target_tokens, args.topk_by_norm)
    target_acts = acts[idx]
    mode = "top-k by norm" if args.topk_by_norm else "last-k"

    if reference is None:
        print(f"(layer-{TARGET_LAYER} acts {tuple(prim_acts.shape)}; "
              f"injecting {target_acts.shape[0]} pos [{mode}])\n")
    else:
        rel = (acts[idx].norm(dim=-1)
               / prim_acts[idx].norm(dim=-1).clamp_min(1e-6)).mean().item()
        print(f"(diff {MODELS[primary]['name']} - {MODELS[reference]['name']}; "
              f"prompt len {prim_acts.shape[0]}; injecting {target_acts.shape[0]} pos "
              f"[{mode}]; mean ||diff||/||primary|| = {rel:.3f})\n")

    answers = run_oracle_on_acts(
        env, target_acts, question,
        max_new_tokens=args.oracle_max_new_tokens,
        coeff=args.steering_coeff,
        num_samples=args.oracle_samples,
        temperature=args.oracle_temp,
    )
    print(f"[oracle answers | {len(answers)} draw(s) @ T={args.oracle_temp}]")
    for j, a in enumerate(answers, 1):
        print(f"  {j}. {a}")
    verdicts = [classify_yesno(a) for a in answers]
    yes, no = verdicts.count("yes"), verdicts.count("no")
    unclear = verdicts.count("unclear")
    parseable = yes + no
    if parseable:
        print(f"  yes-rate: {yes}/{parseable} ({yes / parseable:.0%})   "
              f"[yes={yes} no={no} unclear={unclear}]\n")
    else:
        print(f"  yes-rate: n/a   [yes=0 no=0 unclear={unclear}]\n")
    print("-" * 60)


def repl(env, args, key):
    """Prompt/answer + oracle loop for the chosen assistant `key`. Raises
    KeyboardInterrupt (Ctrl+C) to bubble back up to model selection."""
    name = MODELS[key]["name"]
    print("\n" + "=" * 60)
    print(f"INTERACTIVE ORACLE  (assistant {key} = {name})")
    print("=" * 60)
    print("Type a prompt for the assistant, then a question for the oracle.")
    print("Empty prompt skips a round; Ctrl+C returns to model selection.\n")

    while True:
        prompt = input("assistant prompt> ").strip()
        if not prompt:
            continue
        question = input("oracle question> ").strip()
        if not question:
            print()
            continue
        run_round(env, args, key, prompt, question)


def parse_script(path):
    """Parse a batch script into a list of (primary, reference, assistant_prompt,
    oracle_prompt). `reference` is None for single-model requests.

    Format: repeating triples of non-empty lines —
        <primary digit> [reference digit]   # 2nd digit => diff acts(primary)-acts(reference)
        <assistant prompt>
        <oracle prompt>
    Blank lines and lines starting with '#' are ignored, so triples may be
    separated by blank lines or annotated with comments for readability."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    lines = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]

    if len(lines) % 3 != 0:
        raise ValueError(
            f"Script must contain triples (digit / assistant prompt / oracle prompt); "
            f"got {len(lines)} non-empty lines, which is not a multiple of 3."
        )

    requests = []
    for i in range(0, len(lines), 3):
        digits = lines[i].split()
        if not 1 <= len(digits) <= 2:
            raise ValueError(
                f"Request {i // 3 + 1}: first line must be one digit (single-model) or "
                f"two digits '<primary> <reference>' (diff); got {lines[i]!r}."
            )
        for d in digits:
            if d not in MODELS:
                raise ValueError(
                    f"Request {i // 3 + 1}: invalid model digit {d!r}. "
                    f"Choose one of: {', '.join(MODELS)}."
                )
        primary = digits[0]
        reference = digits[1] if len(digits) == 2 else None
        requests.append((primary, reference, lines[i + 1], lines[i + 2]))
    return requests


def run_script(env, args, path):
    """Execute every request in `path` in order, printing each result."""
    requests = parse_script(path)
    print("\n" + "=" * 60)
    print(f"BATCH SCRIPT: {path}  ({len(requests)} request(s))")
    print("=" * 60)

    for n, (primary, reference, prompt, question) in enumerate(requests, start=1):
        if reference is None:
            label = f"model {primary} = {MODELS[primary]['name']}"
        else:
            label = (f"diff {primary}-{reference} = "
                     f"{MODELS[primary]['name']} - {MODELS[reference]['name']}")
        print("\n" + "#" * 60)
        print(f"# REQUEST {n}/{len(requests)}  —  {label}")
        print("#" * 60)
        print(f"assistant prompt> {prompt}")
        print(f"oracle question>  {question}")
        run_round(env, args, primary, prompt, question, reference)

    print("\nDone.")


def main():
    args = parse_args()
    env = load_oracle()

    if args.script:
        run_script(env, args, args.script)
        return

    while True:
        key = select_model(env)
        if key is None:
            break
        try:
            repl(env, args, key)
        except KeyboardInterrupt:
            print("\n\n[Ctrl+C] returning to model selection...")

    print("\nDone.")


if __name__ == "__main__":
    main()

# %%
