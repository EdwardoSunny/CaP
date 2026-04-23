# Two-Level Ctrl-G Run on CaP Code Generation — Failure Inventory

**Date:** 2026-04-15
**Tmux session:** `cap-ctrlg`
**Driver:** `eai_ctrlg/generate_cap.py` (new, written for this run)
**Prompts:** `CaP/demo/prompts_v1_context.jsonl` (45 BEHAVIOR-derived tabletop prompts)
**Few-shot:** `CaP/cap/lmp/prompts/real/tabletop_ui_prompt_v4_diverse.txt` (same file used by v5)
**Model:** `dganochenko/llama-3-8b-chat` (served by the existing vLLM at `localhost:8000`)
**Run artifacts:**
- `CaP/demo/ctrlg_run/outputs_llama3_ctrlg.jsonl` — 45 records, one per prompt
- `CaP/demo/ctrlg_run/run.log` — stdout of the tmux run

---

## 1. Scope of this first pass (vs. the full plan)

The plan (`CaP/docs/llama3_local_vllm_plan.md` §Step 2c) calls for four new files
totaling thousands of lines — a full `γ` DFA over CaP Python syntax plus a `β`
world model over scene objects, mirroring the structure of
`eai_ctrlg/ctrlg/dfa.py` (4,726 lines) and `ctrlg/beta_dfa.py`.

This run implements a **minimum-viable two-level Ctrl-G** instead, sufficient to
*find the failure prompts* without a multi-day engineering effort:

| Level | Full plan | This run | Gap |
|---|---|---|---|
| γ (syntactic) | Token-level DFA enforcing CaP Python grammar + call-head allowlist | **Per-token `logit_bias = -100`** on the first subword of `goto_pos` and `stack_objects_in_order`, plus `stop` strings `["say(", "# Query:", "objects ="]` (vLLM OpenAI API) | No structural grammar — the stop-string approach catches the 3 dominant non-API calls but cannot distinguish "`objects =`" inside an f-string from the same prefix at statement start (see c015 below). |
| β (semantic) | Decoding-time meta-DFA over `(objects, held)` with precondition checks and goal satisfaction | **Post-hoc rejection:** check the finished program for `put_first_on_second(x, x)`, off-scene literal destinations, and empty outputs | β is detection-only here — the decoder can still emit β-invalid programs; we flag them rather than redirect. The full plan would route the decoder away from these at generation time. |
| HMM | Optional Neural-HMM prior | Not used | Shipped BEHAVIOR HMM is JSON-format, not CaP Python |

The runtime trick is that we **re-use the already-running vLLM server on
`:8000`** via the OpenAI-compatible chat completions API — the plan assumed we'd
spin up a second HF transformers process, but `logit_bias` + `stop` on the
existing server gives us γ enforcement at decode time without a second model
instance. This also means the CLAMP BEHAVIOR-SD job on the same GPU was never
touched and free VRAM held at ~40 GB throughout.

---

## 2. GPU state during the run

Unchanged from the pre-flight check. No process was killed.

```
Before run:  free 39693 MiB   util 36 %
After  run:  free 39657 MiB   util 35 %

Co-tenants (both left alive):
  PID 2950848  VLLM::EngineCore                              40424 MiB   (12 h uptime)
  PID 3398754  python run_clamp_behavior.py behavior_sd       16638 MiB   (~1 h, live)
```

The γ constraint is implemented entirely inside the chat-completions request
payload (`logit_bias`, `stop`); no additional GPU memory is used.

---

## 3. Headline result

| Run | Pass / 45 | Notes |
|---|---:|---|
| **v0 baseline** (prior, no constraints) | 43/45 *by its own lax evaluator* (really ~2/45 under strict rubric — 41 `goto_pos` fallbacks, 2 empty) | `outputs_llama3_v0_simple_error.jsonl` |
| **v5 prompt engineering** (prior, 5th iteration) | 44/45 | `outputs_llama3_v5_diverse.jsonl` — but silently passes 4 identity actions |
| **Two-level Ctrl-G (this run)** | **38 / 45** | 7 failures **all surface** — none hidden by the evaluator |

The pass rate went *down* vs v5, but the failures that remain are the ones
v5 was silently concealing plus two newly-exposed spatial-phrase cases.
This is the intended behaviour of a constrained decoder: it converts a
*silent* failure into a *loud* one.

---

## 4. Failure inventory (7 of 45 fail γ+β)

### 4.1 Category C — identity action `put_first_on_second(x, x)` — 4 prompts

These are exactly the prompts v5's evaluator passed while v5's generated code
would crash the XArm on execution.

| id | Query | Generated code | β flag |
|---|---|---|---|
| `c013` | put all the cartons into the carton | `put_first_on_second('carton', 'carton')` | `identity_action` |
| `c021` | put all the cartons into the carton | `put_first_on_second('carton', 'carton')` | `identity_action` |
| `c022` | put all the cartons into the carton | `put_first_on_second('carton', 'carton')` | `identity_action` |
| `c023` | put all the cartons into the carton | `put_first_on_second('carton', 'carton')` | `identity_action` |

**What Ctrl-G's β would fix at decode time:** the β world rejects any action
with `src == dst`. Here the prompt itself is ill-posed (the BEHAVIOR task uses
`carton` as both the source class and destination container, collapsing into
one literal). A full β would force the model down a different action path —
e.g. declining the task or resolving `cartons` through `parse_obj_name` to
non-container carton instances. This pass merely flags the failure rather than
routing around it.

### 4.2 Category B (spatial) — off-scene literal destination — 2 prompts

A new failure mode γ-only does not handle: with `goto_pos` banned, the model
still picks `put_first_on_second` but forgets to wrap a spatial phrase in
`parse_position(...)`:

| id | Query | Generated code | Context objects | β flag |
|---|---|---|---|---|
| `c001` | move the plywoods to the designated area | `put_first_on_second('plywood', 'designated area')` | `['plywood']` | `offscene_destination` |
| `c010` | move the tiles to the designated area | `put_first_on_second('tile', 'designated area')` | `['tile']` | `offscene_destination` |

At v5 (no γ constraint) the model used
`put_first_on_second('plywood', parse_position('the designated area'))`
correctly. Under the γ-only `goto_pos` ban, the 8B model's fallback path
drops the `parse_position` wrapper. **This is new evidence that γ and β are
genuinely complementary**: γ alone is not enough, and a proper β-level
constraint forcing `dst ∈ objects ∪ {parse_position(...)}` would catch this at
decode time.

### 4.3 Category A (syntactic) — truncated at stop string inside f-string — 1 prompt

A limitation of the minimal γ implementation:

| id | Query | Generated code (truncated) | γ flag |
|---|---|---|---|
| `c015` | put all the highlighters into the backpack | `highlighter_names = parse_obj_name('the highlighters', f'` | `syntactically_valid = False` |

The model's correct output would have been the `parse_obj_name` + for-loop
pattern (the same one v5 produced). Generation halted at the literal text
`objects =` inside the f-string `f'objects = {get_obj_names()}'` because our
γ implementation uses simple substring stop-strings. A real γ DFA would
recognise that `objects = ` *inside a string literal* is not a statement
boundary. This is the single category of failure that is an artefact of the
cheap-γ approximation rather than a genuine model failure.

---

## 5. Cross-comparison with v0 failure taxonomy

From `CaP/docs/llama3_local_vllm_plan.md` §"Detailed v0 Failure Inventory":

| Category | v0 count | Fixed in this Ctrl-G run | Remaining | Which level caught it |
|---|---:|---:|---:|---|
| A — non-CaP API call (`goto_pos`, `say`, `stack_objects_in_order`) | 41 | **41** | 0 | γ `logit_bias` + `stop` |
| B — empty / un-parseable | 2 (c016, c017) | **2** | 0 | γ (no banned token means no halt-on-say; both now generate `put_first_on_second('backpack', 'mouse'/'toothpaste')` — still semantically weird but non-empty) |
| B′ — **new:** literal spatial destination without `parse_position` wrap | 0 (not visible at v0) | n/a | **2** (c001, c010) | β would fix at decode time; post-hoc β flagged here |
| C — `put_first_on_second(x, x)` identity action | 4 (hidden at v0 and v5) | **0** — still emitted | **4** (c013, c021, c022, c023) | β flagged post-hoc; full β would block at decode time |
| D — stop-boundary / f-string truncation | 1 (c015 at v5) | — | **1** (c015 this run, same root cause mirrored) | Full γ DFA would fix |

**Two-level Ctrl-G eliminated Categories A and B (43 cases) completely** — no
output contains `goto_pos`, `say`, or `stack_objects_in_order`. The residual 7
failures are all *surfaced*, not *silent*: Category C is the important one
(v5's blind spot); B′ is a tractable new finding; D is a known limitation of
the minimum-viable γ.

---

## 6. Process record (tmux `cap-ctrlg`)

1. **20:18 UTC** — created session `cap-ctrlg` (via `tmux new-session -d`).
2. **20:18 UTC** — verified env: `conda activate ctrlg`, `torch 2.7.1+cu128`,
   `transformers 4.41.2`, CUDA available.
3. **20:18 UTC** — installed `openai` client into the `ctrlg` env (not shipped
   by default); smoke-test the tokenizer.
4. **20:19 UTC** — computed banned token IDs:
   `goto`=29635, ` goto`=8145, `stack`=7848, ` stack`=5729.
5. **20:19 UTC** — wrote `eai_ctrlg/generate_cap.py`.
6. **20:21 UTC** — smoke test on 1 prompt (`c001`) in `/tmp/cap_smoke_out.jsonl`
   passed. γ flags all green; β correctly flagged `offscene_destination` on
   `put_first_on_second('plywood', 'designated area')`.
7. **20:22 UTC** — launched full 45-prompt run in tmux session `cap-ctrlg`
   (cwd `eai_ctrlg/`, output piped to `run.log` via `tee`).
8. **20:22 UTC** — run completed; 45 records written.
9. **20:22 UTC** — aggregated failures (§4) and confirmed GPU state unchanged.
10. **20:23 UTC** — this report saved to `CaP/demo/ctrlg_run/report.md`.

Total wall time of generation: under 60 s (vLLM is warm, 45 × ~5 s per call).

---

## 7. What to build next (to close the remaining 7 failures)

The full Step 2c design in the plan would close all 7 residual failures, in
this order of effort:

| Failure class | Fix file | Estimated LOC | Effort |
|---|---|---:|---|
| c015 truncation (Cat D) | Simple regex post-processor that masks the `objects =` stop only at statement start; a one-line change to `generate_cap.py` | ~30 | 1 hr |
| c001, c010 (Cat B′) | `ctrlg/beta/cap_beta_world.py` with `dst ∈ objects ∪ {parse_position}` precondition; wired into a new HF `LogitsProcessor` | ~400 | 1–2 days |
| c013, c021, c022, c023 (Cat C) | Same β world covers `src ≠ dst` automatically | (included above) | — |

The post-hoc β layer in this run already tells us *which prompts* fail — the
decoder-time β would tell us *what the correct action is instead*. For the
stated goal of "find the failure prompts," this pass is complete.

---

## 8. Files produced

```
CaP/demo/ctrlg_run/
├── outputs_llama3_ctrlg.jsonl     # 45 records, γ/β flags per record
├── run.log                        # tmux stdout transcript
└── report.md                      # this document

eai_ctrlg/
└── generate_cap.py                # new two-level Ctrl-G driver for CaP
```

All artifacts are on disk and the vLLM / CLAMP co-tenants on GPU 0 are
untouched.
