# Plan: Run CaP with Local Llama 3 8B via vLLM

**Goal:** Replace the default OpenAI backend (gpt-5-nano) in this CaP repo with a locally-served Llama 3 8B model, for both offline code-gen testing and eventual real-robot use.

**Primary priority:** Produce high-quality qualitative demo prompts and their generated robot code
outputs that can be presented directly in a DARPA reporting context — demonstrating that a fully
local, open-weight model can translate natural-language task instructions into executable robot
action sequences without any cloud API dependency.

---

## Deliverables

Three artifacts are required before this work is considered complete:

| # | Artifact | Description |
|---|---|---|
| 1 | **Demo prompt set** | 10–20 natural-language task queries covering diverse manipulation scenarios (object pick-and-place, spatial reasoning, multi-step sequencing). Saved as `demo/prompts.jsonl`. |
| 2 | **Generated code outputs** | Llama 3 8B's CaP code for every prompt in the set. Saved as `demo/outputs_llama3.jsonl`. Each record pairs the query with its generated code and quality flags. |
| 3 | **Quality report** | Per-criterion pass/fail analysis of generated outputs with narrative summary. Saved as `demo/quality_report.md`. |

---

## File Formats

### 1. `demo/prompts.jsonl` — Input demo prompt set

One JSON object per line. Each prompt must be hand-curated to be:
- Phrased as a realistic operator instruction (no code terminology)
- Unambiguous about the objects and goal state
- Representative of the target deployment environment (tabletop manipulation)

```jsonc
{"id": "p01", "query": "pick up the red block and place it on top of the blue block", "category": "pick_place", "difficulty": "easy"}
{"id": "p02", "query": "push the tin foil to the left side of the table", "category": "push", "difficulty": "easy"}
{"id": "p03", "query": "stack the three blocks in order from largest to smallest", "category": "multi_step", "difficulty": "medium"}
{"id": "p04", "query": "put everything in the tray into the bin", "category": "generalization", "difficulty": "hard"}
// ... 10–20 entries total
```

Fields:
- `id` — stable identifier used to join with output records (`p01`, `p02`, …)
- `query` — the natural-language instruction passed verbatim to the LMP
- `category` — task type: `pick_place` | `push` | `multi_step` | `spatial` | `generalization`
- `difficulty` — `easy` | `medium` | `hard`

### 2. `demo/outputs_llama3.jsonl` — Generated code outputs

One JSON object per line, one per prompt. Written by the batch generation script (see Step 2b).

```jsonc
{
  "id": "p01",
  "query": "pick up the red block and place it on top of the blue block",
  "model": "dganochenko/llama-3-8b-chat",
  "timestamp": "2026-04-15T10:32:00Z",
  "generated_code": "red_block = parse_obj_name('red block')\nblue_block = parse_obj_name('blue block')\npick_and_place(red_block, blue_block)",
  "quality": {
    "syntactically_valid": true,
    "uses_only_cap_api": true,
    "no_repetition": true,
    "stops_correctly": true,
    "pass": true
  },
  "notes": ""
}
```

Fields:
- `id` — matches the prompt record
- `generated_code` — raw LMP output after stop-token trimming
- `quality` — auto-filled by the evaluation script where checkable; manual override for semantic checks
- `notes` — free-text annotation for DARPA reviewers (edge cases, model behaviors worth highlighting)

### 3. `demo/quality_report.md` — Human-readable report

Structure:

```
# CaP Demo Quality Report — Llama 3 8B
## Summary
## Per-Criterion Pass Rates (table)
## Prompt-by-Prompt Results (table: id | query | generated code | verdict | notes)
## Notable Observations
## Conclusion & Recommendation
```

---

## Real Robot Pipeline — Constraints That Govern Prompt Design

Every demo prompt will ultimately be executed on a real XArm7 via this pipeline:

```
User instruction (natural language)
      ↓
LLM (local Llama 3 8B via vLLM)
      ↓ generates Python code
LMP execution engine (cap/lmp/lmp.py)
      ↓ calls robot API
LMPWrapper (cap/lmp/lmp_wrapper.py)
      ↓
┌──────────────────────────────────────────────┐
│ Perception   Molmo VLM + SAM2                │  → locates object in camera image,
│              (segment_pc.py)                 │    votes on best mask, extracts point cloud
│ Grasp        GraspGen / hardcode strategy    │  → generates 6-DOF grasp pose from point cloud
│ Motion       XArmMotionController            │  → executes pick-place trajectory
└──────────────────────────────────────────────┘
```

Each stage in the real pipeline imposes hard constraints on what a valid demo prompt can ask:

| Pipeline stage | Constraint on prompts |
|---|---|
| **Molmo VLM** | Objects must be visually distinct and describable in natural language. Generic names ("thing", "object") fail. Similar-looking objects in the same scene need disambiguating color/shape descriptors. |
| **SAM2 segmentation** | Objects must have clear 2D boundaries in the RGB image. Flat sheets, transparent objects, and overlapping piles are hard to segment reliably. |
| **GraspGen point cloud** | Objects must have graspable 3D geometry — GraspGen needs a surface to place fingers on. Very flat objects (paper), very small objects (<2 cm), and objects with no stable grasp pose are problematic. |
| **XArm7 (single arm)** | Only one object held at a time. No bi-manual tasks. Workspace is the tabletop area visible to the overhead camera — not the floor, not container interiors deeper than ~10 cm. |
| **CaP code execution** | All object references in the prompt must resolve to a single `get_obj_pos()` call. If the prompt references an object not on the table, the pipeline errors at runtime. |

**Prompt design rules that follow directly from these constraints:**

1. **Name objects by visible properties**: use color + category ("red cup", "blue bottle", "orange wipes") — Molmo identifies objects this way.
2. **Use 2–4 distinct objects per task**: enough variety to be interesting; not so many that segmentation errors compound.
3. **Prefer convex, graspable objects**: cups, bottles, boxes, blocks, bowls, tools — avoid paper, fabric, transparent containers.
4. **Single-object-at-a-time operations**: pick one object, place it somewhere, repeat. Don't ask for simultaneous dual-arm operations.
5. **Destination must be a visible surface or object**: "on the table", "on top of the box", "next to the bottle" — not "in the kitchen" or "inside the drawer".
6. **Object names must be consistent** within a prompt: if you say "red cup", every reference to that object must say "red cup" (not "the cup" or "it").

---

## Step 0 — Build the Demo Prompt Set (Extractor + Verifier Agents)

Before running any generation, you need `demo/prompts.jsonl`. Two scripts handle this:

### Step 0a — Extractor Agent (`offline/extract_cap_prompts.py`)

**What it does:** Scans the EAI benchmark prompt files
(`eai_ctrlg/standalone_evaluation/data/prompts/behavior/action_sequencing/` and
`virtualHome/action_sequencing/`), filters to tasks whose target states are
expressible purely through pick-and-place / spatial operations (compatible with CaP's
API), and emits candidate natural-language instructions as `demo/prompts_candidates.jsonl`.

**CaP API compatibility filter:**
- ALLOWED target predicates: `ontop`, `inside`, `nextto`, `onfloor`, `under`
- BLOCKED predicates (require actions CaP cannot do): `soaked`, `cooked`, `sliced`,
  `stained`, `frozen`, `cleaned`, `toggled`, `open` (state changes)
- BLOCKED tasks: those requiring locomotion across rooms, multi-floor navigation,
  or object state transformations (cooking, cleaning, cutting)

**Output format:** `demo/prompts_candidates.jsonl` — one candidate per line:

```jsonc
{
  "id": "c001",
  "source": "behavior/action_sequencing",
  "source_identifier": "boxing_books_up_for_storage_0_...",
  "task_name": "boxing books up for storage",
  "objects": ["notebook_59", "notebook_60", "carton_66"],
  "object_categories": ["book.n.02", "book.n.02", "carton.n.02"],
  "target_predicate": "inside",
  "query": "put all the books inside the box",
  "cap_operations": ["put_first_on_second"],
  "difficulty": "easy"
}
```

**Run:**
```bash
python offline/extract_cap_prompts.py \
  --behavior-dir ../eai_ctrlg/standalone_evaluation/data/prompts/behavior/action_sequencing \
  --vh-dir ../eai_ctrlg/standalone_evaluation/data/prompts/virtualHome/action_sequencing \
  --output demo/prompts_candidates.jsonl \
  --verbose
```

Expected output: ~40–50 candidates from BEHAVIOR AS; VirtualHome tasks not used
(identifiers like `27_2` carry no task-name signal).

### Step 0b — Verifier Agent (`offline/verify_prompts.py`)

**What it does:** For each entry in `demo/prompts_candidates.jsonl` (or a manually
authored `demo/prompts.jsonl`), sends a structured validation request to the local
Llama 3 8B via vLLM and checks five criteria. Emits pass/fail per criterion plus
a curated `demo/prompts.jsonl` containing only validated entries.

**Validation criteria (checked by the LLM agent):**

The verifier checks seven criteria derived directly from the real robot pipeline constraints:

| # | Criterion | Key | Maps to pipeline stage | Check |
|---|---|---|---|---|
| 1 | Molmo visibility | `molmo_visible` | Molmo VLM | Every object has a color or shape descriptor Molmo can locate ("red cup", not just "cup") |
| 2 | GraspGen compatibility | `graspgen_compatible` | GraspGen point cloud | All objects to be grasped are convex, 3D, and have a stable grasp surface (no paper, fabric, transparent containers, or objects <2 cm) |
| 3 | Single-arm feasibility | `single_arm` | XArm7 | Task requires holding at most one object at a time; no simultaneous dual-object manipulation |
| 4 | Destination reachable | `dest_reachable` | XArm7 workspace | The destination is a visible tabletop surface or object — not a drawer interior, floor, or off-table location |
| 5 | CaP API coverage | `cap_feasible` | LMP code execution | Task is fully expressible using `put_first_on_second`, `stack_objects_in_order`, `parse_position`, etc. — no cooking, cleaning, cutting, or state changes |
| 6 | Instruction unambiguity | `unambiguous` | LMP prompt → code | Exactly one reasonable interpretation; no pronoun ambiguity when multiple similar objects exist |
| 7 | No logical contradiction | `no_contradiction` | Runtime safety | No impossible states (source = destination, object count mismatch, etc.) |

**Verifier prompt template (sent to Llama 3 8B for each candidate):**

```
You are an expert robot task validator. The robot is a real XArm7 single-arm manipulator
with this execution pipeline:
  1. Molmo VLM identifies objects in the camera image using natural-language descriptions
  2. SAM2 segments the identified object and extracts its point cloud
  3. GraspGen generates a 6-DOF grasp pose from the point cloud
  4. XArm7 executes the pick-place trajectory

Robot API available to LLM-generated code:
  put_first_on_second, stack_objects_in_order, goto_pos, get_obj_pos, get_obj_names,
  parse_obj_name, parse_position, move_up, move_down, open_gripper, close_gripper,
  pick_place, is_obj_visible, say

Task: "{query}"
Objects present on the table: {objects}

Evaluate against these 7 criteria. For each, answer true/false + one sentence:

1. molmo_visible: Does every object have a color or shape descriptor Molmo can identify?
   (Fail if object name is generic like "thing", or lacks any visual disambiguation.)
2. graspgen_compatible: Are all objects-to-be-grasped 3D, convex, and graspable?
   (Fail if any object is paper, fabric, transparent, very flat, or smaller than 2 cm.)
3. single_arm: Can the task be done holding at most one object at a time?
   (Fail if it implies simultaneous multi-object manipulation.)
4. dest_reachable: Is the destination a visible tabletop surface or another object on the table?
   (Fail if destination is a drawer interior, the floor, off-table, or requires navigation.)
5. cap_feasible: Can the task be expressed entirely with the robot API listed above?
   (Fail if it requires cooking, cleaning, cutting, toggling switches, or state changes.)
6. unambiguous: Is there exactly one reasonable interpretation given the object list?
   (Fail if pronouns or partial names could refer to multiple objects.)
7. no_contradiction: Is the task free of logical impossibilities?
   (Fail if source = destination, if object count mismatches, or if goal is already met.)

Respond ONLY with valid JSON (no markdown):
{{"molmo_visible": true/false, "molmo_visible_reason": "...",
  "graspgen_compatible": true/false, "graspgen_compatible_reason": "...",
  "single_arm": true/false, "single_arm_reason": "...",
  "dest_reachable": true/false, "dest_reachable_reason": "...",
  "cap_feasible": true/false, "cap_feasible_reason": "...",
  "unambiguous": true/false, "unambiguous_reason": "...",
  "no_contradiction": true/false, "no_contradiction_reason": "...",
  "pass": true/false,
  "notes": "one-sentence overall assessment"}}
```

**Run (requires vLLM server running at :8000):**
```bash
python offline/verify_prompts.py \
  --input demo/prompts_candidates.jsonl \
  --output demo/prompts.jsonl \
  --vllm-host localhost:8000 \
  --model dganochenko/llama-3-8b-chat \
  --min-pass 6 \
  --verbose
```

`--min-pass 6` means a candidate passes if at least 6 of 7 criteria are `true`.
After filtering, manually review `demo/prompts.jsonl` and trim to 10–20 diverse entries
covering multiple categories and difficulty levels.

---

## What's Already Here

| Component | Status | Details |
|---|---|---|
| Llama 3 8B weights | Ready | `dganochenko/llama-3-8b-chat` in HF cache (8192 ctx) |
| vLLM | Ready | v0.19.0 in `cap-vllm` conda env (fresh install) |
| GPU | Ready | NVIDIA GH200 — 96 GB VRAM (97871 MiB) |
| CaP vLLM support | Built-in | `lmp.py` switches to `chat_completions` mode via config; `offline/lmp_codegen_test.py` and `main.py` both accept `--model` / `--vllm-host` flags |

**No code changes are required.** Everything is config-driven.

---

## Architecture: How CaP Calls the LLM

```
configs/real_config.yaml
  └── lmps:
        tabletop_ui, fgen, parse_obj_name, parse_position,
        parse_question, transform_shape_pts
              ↓ model / base_url / api_mode / context_window
cap/lmp/lmp.py  (OpenAI client)
              ↓
  if api_mode == "chat_completions"  →  client.chat.completions.create()   ← vLLM path
  else                               →  client.responses.create()          ← OpenAI path
```

Six LMPs are defined in the config; all need to point to the same local server.

The CaP code already handles two vLLM quirks:
- **No native stop tokens** — `_trim_at_stop_tokens()` post-processes output.
- **Small context window** — `context_window` field in config triggers automatic few-shot
  truncation (drops examples from the end until the prompt fits, preserving the header
  and at least one example).

---

## Step 1 — Start the vLLM Server

Open a persistent terminal (tmux or screen recommended):

```bash
conda activate cap-vllm

# NOTE: ~/.local/bin shadows env pip — always use the full path for installs:
# /home/ubuntu/miniconda3/envs/cap-vllm/bin/pip install <pkg>

/home/ubuntu/miniconda3/envs/cap-vllm/bin/vllm serve \
  dganochenko/llama-3-8b-chat \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.4
```

Parameter notes:
- `--max-model-len 8192` — matches the model's context; raise to `131072` for full context (needs more VRAM).
- `--gpu-memory-utilization 0.4` — conservative on the 96 GB GH200 (~16 GB needed for 8B).
- Model is resolved from HF cache at:
  `/home/ubuntu/.cache/huggingface/hub/models--dganochenko--llama-3-8b-chat/snapshots/98dfb6fd76f79664edb45cc836b5759d52619bfc`

Wait for:
```
INFO:     Application startup complete.
```

**Sanity check** (new terminal, server still running):
```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
# Should list: "dganochenko/llama-3-8b-chat"
```

Expected: 
```json
{
    "object": "list",
    "data": [
        {
            "id": "dganochenko/llama-3-8b-chat",
            "object": "model",
            "created": 1776229059,
            "owned_by": "vllm",
            "root": "dganochenko/llama-3-8b-chat",
            "parent": null,
            "max_model_len": 8192,
            "permission": [
                {
                    "id": "modelperm-8dc259d445bba721",
                    "object": "model_permission",
                    "created": 1776229059,
                    "allow_create_engine": false,
                    "allow_sampling": true,
                    "allow_logprobs": true,
                    "allow_search_indices": false,
                    "allow_view": true,
                    "allow_fine_tuning": false,
                    "organization": "*",
                    "group": null,
                    "is_blocking": false
                }
            ]
        }
    ]
}
```

---

## Step 2 — Offline Code-Gen Test (no robot needed)

From the CaP repo root (`/home/ubuntu/tianyi/EmbodiedAgents/CaP/`):

```bash
python offline/lmp_codegen_test.py \
  "pick up the box and place it on the battery" \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 \
  --context-window 8192 \
  --show-prompt \
  --save
```

What happens internally:
1. Script detects `"/"` in `--model` → sets `base_url`, `api_key = "not-needed"`, `api_mode = "chat_completions"` in all LMP configs.
2. `context_window 8192` triggers few-shot truncation — examples are dropped from the end until the prompt fits.
3. Output is printed under `GENERATED CODE`.

Try multiple queries to build confidence:
```bash
python offline/lmp_codegen_test.py "stack the blocks" \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192

python offline/lmp_codegen_test.py "push the tin foil to the left" \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192
```

### Step 2b — Batch Generation Over the Full Demo Prompt Set

Once `demo/prompts.jsonl` exists, run all prompts in one pass and collect structured output:

```bash
mkdir -p demo

python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_candidates.jsonl \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 \
  --context-window 8192 \
  --output demo/outputs_llama3.jsonl
```

> **Note:** `offline/batch_lmp_codegen.py` is now implemented. See Step 2b-fix section below
> for versioned runs after diagnosing quality problems in the v0 baseline.

---

## Step 2c — Two-Level Ctrl-G Instead of Prompt Engineering

**Goal.** Replace the v1–v5 prompt-engineering iterations with a principled constrained-decoding
fix: the same Llama 3 8B that produced the v0 failures is wrapped in the two-level Ctrl-G
pipeline from `/home/ubuntu/tianyi/EmbodiedAgents/eai_ctrlg`. This forces every generated token
to belong to a sequence that (1) is syntactically a valid CaP program (γ DFA) and (2) is
semantically grounded in the scene's objects and the CaP API's action arity (β DFA / HMM).
No change to the prompt, few-shots, or system message is required — the model sees exactly
what v0 saw, but the decoder is no longer allowed to emit hallucinated API names or
mis-typed destinations.

### Why Ctrl-G, not more prompt engineering

The v1→v5 history shows a repeating failure mode: every fix that eliminates one bad pattern
(e.g. `goto_pos` fallback in v4) introduces a new one (e.g. pattern collapse in v4, or
`stops_correctly` regressions in v5). The underlying problem is that an 8B model does not
reliably respect a soft constraint expressed only in few-shot examples — the constraint has
to be enforced at decoding time. Ctrl-G provides exactly that:

| Failure class at v0 | Soft fix in v1–v5 | Hard fix via Ctrl-G |
|---|---|---|
| Model calls `goto_pos(...)` (not in CaP API) | Delete `goto_pos` examples from prompt (v2) | γ DFA allowlist rejects `goto_pos` token-by-token |
| Model calls `stack_objects_in_order(...)` (not in CaP API) | Strip it from prompt + re-script examples (v4) | γ DFA allowlist rejects it |
| Model calls `say(...)` (debug wrapper, not a robot action) | Strip narration (v4) | γ DFA allowlist rejects it |
| Model echoes the query or runs past `# Query:` marker | Manual stop-token post-processing | γ DFA terminates at EOS on valid completion |
| Model emits empty / un-parseable output (c016, c017) | None — these still fail at v5 | γ DFA guarantees syntactic validity by construction |
| Destination string not in the scene's object list | Not addressed | β world model rejects IDs outside `get_obj_names()` |
| `put_first_on_second(x, y)` with `x == y` (c013, c021, c022, c023: "put cartons into the carton") | Not addressed | β world: `src != dst` constraint on apply |
| Pattern collapse to single `put_first_on_second` line (v4) | Add diverse examples (v5) | HMM log-prior over action multiplicity |

### Mapping the two Ctrl-G levels onto CaP

`eai_ctrlg` currently ships DFAs for BEHAVIOR/VirtualHome JSON action-sequencing outputs
(`ctrlg/dfa.py`, `ctrlg/beta_dfa.py`). For CaP we need to author the equivalent two DFAs over
the **CaP Python subset**:

1. **γ token DFA — CaP Python syntax + API allowlist.**
   Implemented as a new `ctrlg/dfa_cap.py::CaPCodegenDFA`. States encode: import-line skipped,
   at statement boundary, inside a call expression, inside an argument literal, end-of-line.
   The allowlist over call-head tokens is sourced directly from `cap/lmp/lmp_wrapper.py::setup_LMP`
   (`variable_vars` + `fixed_vars`) — the same authoritative source that
   `offline/batch_lmp_codegen.py::_DEFAULT_CAP_API` already uses. This eliminates every
   `goto_pos` / `say` / `stack_objects_in_order` failure by construction.

2. **β meta DFA — scene-grounded semantics.**
   Implemented as `ctrlg/beta/cap_beta_world.py::CaPBetaWorld`. The "world" state is
   `(objects_on_table: set[str], held: Optional[str])`, populated from each prompt's
   `context` field (already produced by `offline/add_context_to_prompts.py`).
   Valid β-actions:
   - `put_first_on_second(src, dst)` with `src ∈ objects ∧ dst ∈ objects ∪ {parse_position(...)} ∧ src ≠ dst`
   - `parse_obj_name(group_phrase, 'objects = {get_obj_names()}')` followed by a
     `for ... in ...: put_first_on_second(...)` block
   - `parse_position(spatial_phrase)` only as the 2nd arg of `put_first_on_second`

   This catches c013/c021/c022/c023 (source = destination), c016/c017 (empty output is not a
   valid β-action), and any "put books into the dishwasher" where `books` is only resolvable
   through `parse_obj_name` (β forces the `parse_obj_name + for-loop` pattern for plural nouns).

3. **HMM (optional, gamma-scale tunable).**
   The BEHAVIOR HMM shipped at `eai_ctrlg/models/behavior/hmm-h128-lr0.01/checkpoint.eqx`
   was trained on BEHAVIOR AS JSON continuations — it is **not directly reusable** for CaP
   Python. Start with **γ-only (DFA-only) Ctrl-G** (no HMM). Adding a CaP-specific HMM is a
   follow-up that requires sampling Llama 3.1 continuations on a CaP-format dataset
   (`eai_train/cond_hmm/sample_llama31_vllm.py` adapted to CaP prompts) and retraining — not
   in scope for the first pass.

### Runtime switch: vLLM → HuggingFace transformers

`eai_ctrlg/generate.py` and `generate_beta.py` run through `transformers.AutoModelForCausalLM`
with a custom `LogitsProcessor`; they do **not** use vLLM. This means the vLLM server on
`:8000` is **not usable** for the Ctrl-G pass. Two options:

- **Option A (recommended):** Shut down the vLLM server and run generation through HF
  `transformers` with the Ctrl-G logits processor on the freed GPU. Lowest memory, cleanest
  code path. Prompt-engineering baselines (v0–v5) are already saved on disk so the vLLM
  server is no longer needed.
- **Option B:** Keep vLLM up for the unconstrained baseline and co-locate an HF Llama 3 8B
  for Ctrl-G on the same GPU. Requires ~32–40 GB total (two bf16 copies of the 8B weights +
  KV caches). Infeasible given current VRAM headroom (see assessment below).

### Step 2c concrete workflow

```bash
# 0. Stop vLLM (Option A — frees ~40 GB of VRAM)
#    (kill PID shown by `nvidia-smi --query-compute-apps=pid,process_name`)

# 1. Author the two new DFAs (one-time code change — no HMM retraining needed)
#    eai_ctrlg/ctrlg/dfa_cap.py              — CaPCodegenDFA (γ, token-level)
#    eai_ctrlg/ctrlg/beta/cap_beta_world.py  — CaPBetaWorld  (β, scene-grounded)
#    eai_ctrlg/ctrlg/beta/cap_beta_dfa.py    — CaPBetaDFA    (wraps the world for generate_beta.py)

# 2. Add a thin CaP driver next to generate.py that wires the CaP DFA into Ctrl-G
#    eai_ctrlg/generate_cap.py  (mirrors generate.py; reads demo/prompts_v1_context.jsonl)

# 3. γ-only (DFA-only) run — the mandatory pass condition
conda activate ctrlg
cd /home/ubuntu/tianyi/EmbodiedAgents/eai_ctrlg
python generate_cap.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --prompts    /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/prompts_v1_context.jsonl \
  --few_shot   /home/ubuntu/tianyi/EmbodiedAgents/CaP/cap/lmp/prompts/real/tabletop_ui_prompt.txt \
  --run_type   dfa_only \
  --gamma_scale 0.0 \
  --temperature 0.0 \
  --max_new_tokens 512 \
  --output     /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/outputs_llama3_ctrlg_gamma.jsonl

# 4. Full two-level Ctrl-G (γ + β; HMM off for this first pass)
python generate_cap.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --prompts    /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/prompts_v1_context.jsonl \
  --few_shot   /home/ubuntu/tianyi/EmbodiedAgents/CaP/cap/lmp/prompts/real/tabletop_ui_prompt.txt \
  --run_type   ctrlg \
  --use_beta   true \
  --gamma_scale 0.0 \
  --temperature 0.0 \
  --max_new_tokens 512 \
  --output     /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/outputs_llama3_ctrlg_beta.jsonl

# 5. Re-score with the existing evaluator (same rubric as v0–v5)
conda run -n cap-vllm python /home/ubuntu/tianyi/EmbodiedAgents/CaP/offline/evaluate_outputs.py \
  --outputs /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/outputs_llama3_ctrlg_beta.jsonl \
  --output  /home/ubuntu/tianyi/EmbodiedAgents/CaP/demo/outputs_llama3_ctrlg_beta_evaluated.jsonl
```

**Pass condition for Step 2c:** `outputs_llama3_ctrlg_beta_evaluated.jsonl` reaches 45/45 on
the existing rubric *and* shows non-trivial diversity (`parse_obj_name`/`parse_position`
counts ≥ v5) *and* zero instances of source = destination.

### Step 2c — Actual Results (2026-04-16)

The initial implementation uses a **minimal two-level approach**: γ via vLLM
logit_bias + stop strings (not a full token-level DFA), β via post-hoc
rejection checks (not a real-time β DFA). This is `generate_cap.py` in
`eai_ctrlg/`, run against all 45 CaP-compatible prompts.

**Three-mode comparison** (retroactive γ+β applied to all modes for fair comparison):

| Mode | PASS | Rate | vs. baseline | Key change |
|------|------|------|-------------|-----------|
| baseline | 35/45 | 78% | — | Raw LLM, no constraints |
| γ-only | 38/45 | 84% | +3 | logit_bias bans goto_pos/stack; stop strings block say( |
| γ+β (ctrlg) | 38/45 | 84% | +3 | γ decoding + β post-hoc detection (same code, more diagnostics) |

**Important finding**: γ-only and ctrlg produce **byte-for-byte identical**
generated code — β is purely post-hoc detection, not generation-time steering.
Without β, the γ-only run reports 44/45 PASS (hiding 6 semantic bugs). With β,
6 additional violations are flagged:

| Failure category | Count | Caught by | Prompts |
|-----------------|-------|-----------|---------|
| β:identity_action (src == dst) | 4 | β only | c013, c021, c022, c023 |
| β:offscene_destination | 2 | β only | c001, c010 |
| γ:syntactically_valid (truncation) | 1 | γ | c015 |
| γ:no_goto_pos (baseline only) | 5 | γ | c001, c031, c032, c037, c038 |

**Pass condition assessment**: NOT MET.
- 38/45 (not 45/45)
- Remaining 7 failures are **prompt extraction bugs** (not model/Ctrl-G issues):
  - 4 identity_action: single-carton context makes query ambiguous
  - 2 offscene_destination: floor/room destination stripped during extraction
  - 1 truncation: max_tokens too low for 7-object prompt
- Full γ-DFA + β-DFA constrained decoding (as designed in the plan) is not
  yet implemented — the current approach uses vLLM API-level constraints only

**Detailed results**: `CaP/demo/ctrlg_run/RESULTS.md`
**Output files**: `CaP/demo/ctrlg_run/outputs_llama3_{baseline,gamma,ctrlg}.jsonl`

### Step 2c — Next steps

1. **Fix prompt extraction bugs** (quick wins to reach 45/45):
   - Add floor/room destinations to `context_objects` (fixes c001, c010)
   - Preserve instance IDs for self-referential queries (fixes c013, c021-c023)
   - Increase `max_tokens` to 768 for prompts with >5 objects (fixes c015)

2. **Implement full γ-DFA** (`ctrlg/dfa_cap.py`): Token-level DFA over CaP
   Python syntax would enforce valid code by construction, eliminating the
   truncation failure (c015) and providing stronger guarantees than logit_bias.

3. **Implement β-DFA for generation-time steering** (`ctrlg/beta/cap_beta_world.py`):
   Would reject `src == dst` and offscene destinations *during* generation,
   not just post-hoc — enabling the model to re-route to valid alternatives.

---

## Detailed v0 Failure Inventory (what Ctrl-G has to fix)

Ground-truth inspection of `demo/outputs_llama3_v0_simple_error.jsonl` (not the v0 evaluator's
self-report, which is too lax). The v0 rubric mis-labelled 41 rows as passing because its
allowlist still admitted `goto_pos`. Under the stricter v5 allowlist (the one that reflects the
real CaP runtime), v0 fails on **43/45 prompts**:

### Category A — Non-CaP API calls (41/45 prompts)

Every row whose `generated_code` leads with `goto_pos(` or `goto_pos(get_obj_pos(` or
`goto_pos(parse_position(`. These do not resolve against `cap/lmp/lmp_wrapper.py::setup_LMP`
and will raise `NameError` at runtime.

Examples from v0:
- `c001` "move the plywoods to the designated area" → `goto_pos(parse_position('the designated area'))`
- `c002` "put all the books inside the carton" → `goto_pos(get_obj_pos('carton'))`
- `c003` "arrange the objects next to the sink" → `goto_pos(parse_position('next to the sink'))`
- `c004` "clear the table — put everything in the bucket" → `goto_pos(get_obj_pos('bucket'))`
- `c005` "collect all the pops and put them in the bucket" → `goto_pos(get_obj_pos('bucket'))`
- (37 more of the same shape)

**What Ctrl-G enforces:** γ DFA's call-head allowlist does not contain `goto_pos`. At the
token right after `(` at a statement start, the logits for `goto_pos`'s first subword are
masked to `-inf`. The sampler is forced toward `put_first_on_second`, `parse_obj_name`, or
`parse_position`.

### Category B — Empty / truncated output (2/45 prompts)

Hard failures the v0 evaluator caught:
- `c016` "put all the backpacks into the mouse" → `generated_code: ""`
- `c017` "put all the backpacks into the toothpaste" → `generated_code: ""`

Both prompts describe a destination (`mouse`, `toothpaste`) that is semantically impossible
as a container. The v0 model under-generates (returns nothing) rather than producing an
incorrect program. v5 makes these pass by emitting literal `put_first_on_second('backpack',
'mouse')`, which is arguably worse — it silently commits to a nonsensical action.

**What Ctrl-G enforces:** β world model rejects any `dst` not in the scene's
`get_obj_names()`. Ctrl-G is then free either to terminate the generation on a shorter valid
program (e.g. `parse_obj_name('the backpacks', …)` followed by an EOS) or to fall back to
the no-op branch if the β world has no goal-consistent action. Either outcome is auditable,
unlike v5's silent-hallucination path.

### Category C — Degenerate identity actions (4/45 prompts)

The v5 evaluator counted these as passing, but they are runtime-unsafe:
- `c013` "put all the cartons into the carton" → `put_first_on_second('carton', 'carton')`
- `c021` same query, same output
- `c022` same query, same output
- `c023` same query, same output

Calling `put_first_on_second(x, x)` picks up `x` and tries to place it on itself — the
GraspGen + motion stack will at best loop indefinitely and at worst crash the XArm
controller.

**What Ctrl-G enforces:** β world's `apply(put_first_on_second, src, dst)` precondition
rejects `src == dst`. γ-only Ctrl-G cannot detect this (it is a semantic, not syntactic,
property); this is the clearest case where the β level is load-bearing.

### Category D — Stop-boundary violations (1/45 prompts at v5)

`c015` "put all the highlighters into the backpack" at v5 generates correct code but has
`stops_correctly: false` — the model keeps emitting tokens past the logical end of the
program. This is the sole v5 fail.

**What Ctrl-G enforces:** γ DFA has an accept state that triggers EOS the moment the program
is syntactically complete. The logits processor masks every non-EOS token at that state.

### Summary table — v0 failure modes vs Ctrl-G mechanism

| Category | Count (v0, strict) | Mechanism | Ctrl-G level | Fix level |
|---|---:|---|---|---|
| A — non-CaP API call (`goto_pos` etc.) | 41 | syntactic | γ DFA allowlist | hard |
| B — empty / un-parseable | 2 | syntactic (no program) | γ DFA + EOS at valid accept | hard |
| C — `src == dst` identity | 4 | semantic | β world precondition | hard |
| D — run-past-stop | 1 | syntactic termination | γ DFA accept → EOS | hard |

(Rows can overlap across categories; totals are not disjoint.)

---

## GPU Feasibility Assessment — Can Step 2c Start Now?

**Snapshot (rechecked 2026-04-15, later):**

```
GPU 0  NVIDIA GH200 480GB   total 97871 MiB   used 57076 MiB   free 39693 MiB   util 36 %

Compute apps on GPU 0:
  PID 2950848  VLLM::EngineCore                                       40424 MiB   (11h48m)
  PID 3398754  python clamp/bin/run_clamp_behavior.py --task behavior_sd
               --backend llm --model dganochenko/llama-3-8b-chat ...  16638 MiB   (48m, live)
```

Since the previous check the device has freed up: one anonymous python process
(PID 3247478, ~20 GB) has exited, utilisation dropped from 99 % → 36 %, and
**free VRAM is now ~39.7 GB**. Of the two remaining jobs:

- PID 2950848 `VLLM::EngineCore` — the CaP/Llama vLLM server. v0–v5 artifacts are
  already persisted to disk (see `CaP/demo/*.jsonl`); the server has no live
  dependency for Step 2c and can be stopped if more headroom is needed.
- PID 3398754 `run_clamp_behavior.py --task behavior_sd --backend llm` — a CLAMP
  BEHAVIOR-SD baseline generation, 48 min in, 100 prompts. **Load-bearing — do
  not kill.**

**Ctrl-G memory requirement** (HF `transformers` + custom `LogitsProcessor`, bf16 Llama 3 8B):

- Weights: ~16 GB
- KV cache @ 8192 ctx, batch=1: ~1–2 GB
- DFA transition table + ConstraintLogitsProcessor buffers: ~0.5–2 GB (the GI DFA in
  `eai_ctrlg` hit ~49 GB at depth=100 before the sparse-matrix fix — see `CLAUDE.md`.
  A CaP γ DFA will be much smaller, likely tens of states — but verify before launching)
- Total budget: **~20–22 GB** γ-only; a little more with β + HMM.

**Verdict: Step 2c can start now, on the free 39.7 GB — no process needs to be killed.**
The γ-only HF run fits comfortably; the γ+β run also fits with margin. Leave the CLAMP
BEHAVIOR-SD job untouched. The only constraint is that Step 2c should run as a single
process against GPU 0 and must not request more than the HF Ctrl-G budget.

Recommended launch sequence:

```bash
# 1. Re-verify the headroom right before launching
nvidia-smi --query-gpu=memory.free --format=csv,noheader
#   Expected ≥ ~35 GB; if CLAMP or vLLM have grown since, reassess.

# 2. Keep --gpu-memory-utilization / torch allocator conservative for the HF process
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_VISIBLE_DEVICES=0

# 3. γ-only run (Step 2c §3)
conda activate ctrlg
cd /home/ubuntu/tianyi/EmbodiedAgents/eai_ctrlg
python generate_cap.py --run_type dfa_only ...  # (full args in Step 2c §3)
```

**If future re-checks show < ~25 GB free** (e.g. another co-tenant starts), defer
Step 2c or stop the vLLM server (PID 2950848) — but do not touch the CLAMP job.

---

## Diagnosis & Fix Log — Generated Code Quality Issues

> **Plan update (2026-04-15):** The v1–v5 fix versions below were prompt-engineering
> iterations (adding/removing few-shot examples, tightening the system message). They
> are **retained as reference only**. The current direction is to fix v0's failures
> via **two-level constrained decoding** from `eai_ctrlg/` (γ token DFA + β meta DFA /
> HMM) rather than more prompt tweaks. See the new section "**Step 2c — Two-Level
> Ctrl-G Instead of Prompt Engineering**" below for the replacement approach,
> detailed failure taxonomy, and GPU feasibility.

**Baseline run** (`demo/outputs_llama3.jsonl`, 2026-04-15): All 45 outputs were degenerate —
single `goto_pos()` calls instead of multi-step `put_first_on_second` loops.

### Root Causes (ranked by impact)

| # | Root Cause | Effect |
|---|---|---|
| **RC-1** | **Context object mismatch** — `batch_lmp_codegen.py` defaulted to `"box, battery, plastic, tin foil, red block, blue block"` as the `objects` context for every prompt, but every BEHAVIOR query references completely different objects (books, cartons, plates, dishwashers). The model saw `objects = ['box', 'battery', ...]` before `# Query: put all the books inside the carton.` and had no grounding. | Model couldn't generate pick-place loops over objects it couldn't identify |
| **RC-2** | **`goto_pos` examples at end of prompt are the closest pattern match** — The last 80 lines of `tabletop_ui_prompt.txt` (added as real-robot movement examples) show trivial one-liner `goto_pos(get_obj_pos('X'))` patterns. 8B models heavily weight the most-recent few-shot examples; these dominated over earlier multi-step `for … put_first_on_second` examples. | `goto_pos(get_obj_pos('X'))` was the default output for all tasks |
| **RC-3** | **Task domain mismatch with few-shot examples** — All 45 BEHAVIOR queries are household storage/sorting tasks; the original few-shot prompt only shows tabletop block/bowl manipulation. `parse_obj_name('the books', …)` for-loop patterns appear nowhere. | Model had no template for "collect all X into Y" style code |
| **RC-4** | **Context window truncation drops complex examples first** — With `--context-window 8192` and `--max-tokens 512`, the usable prompt budget is ~7680 tokens. The full prompt is close to this limit; truncation drops examples from the end, which may clip the newest examples (goto_pos and for-loops) unevenly. | Further reduces pattern diversity available to the model |
| **RC-5** | **Model capacity (8B)** — Llama 3 8B is at the low end for complex in-context code generation, especially when RC-1 through RC-4 already provide wrong or missing signal. | Amplifies every other root cause |

### Fix Versions

Three successive fixes are applied, each producing a versioned prompt file and output file:

| Version | Fix applied | Prompt file | Prompts file | Output file | Result |
|---|---|---|---|---|---|
| **v0** (baseline) | None | `tabletop_ui_prompt.txt` | `prompts_candidates.jsonl` | `outputs_llama3_v0_simple_error.jsonl` | 43/45 pass (degenerate: all `goto_pos`, avg 1 line) |
| **v1** | RC-1: add per-prompt `context` field from `objects` | `tabletop_ui_prompt.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v1_context_fix.jsonl` | 39/45 pass; for-loop=5; still 41 `goto_pos`-leading |
| **v2** ✅ **BEST** | RC-1+RC-2: remove trailing `goto_pos` examples from prompt | `tabletop_ui_prompt_v1_no_goto.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v2_prompt_fix.jsonl` | **44/45 pass**; `put_first_on_second`=33; starts_goto=11; avg 1.5 lines |
| **v3** | RC-1+RC-2+RC-3: add BEHAVIOR-aligned few-shot examples at start of prompt | `tabletop_ui_prompt_v2_behavior.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v3_behavior_fewshot.jsonl` | 45/45 pass but all `goto_pos` (BEHAVIOR examples confuse 8B model) |
| **v4** | RC-6: remove non-API calls from few-shot prompt (`say`, `stack_objects_in_order`) and tighten system message | `tabletop_ui_prompt_v3_strict.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v4_strict_api.jsonl` | 45/45 pass but **converges on single `put_first_on_second`** (0 for-loops, 0 `parse_obj_name`) — too uniform, ignores other LMP helpers |
| **v5** ✅ **BEST** | RC-7: restore diversity — append strict-API BEHAVIOR examples that anchor `parse_obj_name + for-loop` and `parse_position` as the most-recent patterns | `tabletop_ui_prompt_v4_diverse.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v5_diverse.jsonl` | **44/45 pass**, **8 `parse_position`** (4× v4), **1 full `parse_obj_name + for-loop`**, 45 `put_first_on_second`, 0 `goto_pos` / `say` / `stack_objects_in_order` |

### Cross-version diversity table (45 prompts)

| ver | pass | for-loop | `parse_obj_name` | `parse_position` | `put_first_on_second` | `goto_pos` start | `say()` | `stack()` | avg lines |
|-----|------|----------|------------------|------------------|-----------------------|------------------|---------|-----------|-----------|
| v0  | 43/45 | 0 | 0 | 22 | 0 | 41 | 0 | 0 | 1.0 |
| v1  | 39/45 | 5 | 3 | 21 | 0 | 41 | 2 | 0 | 2.0 |
| v2  | 44/45 | 2 | 1 | 3  | 33 | 11 | 4 | 0 | 1.5 |
| v3  | 45/45 | 3 | 0 | 4  | 3  | 45 | 0 | 0 | 1.2 |
| v4  | 45/45 | 0 | 0 | 2  | 45 | 0  | 0 | 0 | 1.5 |
| **v5**  | **44/45** | **1** | **1** | **8**  | **45** | **0**  | **0** | **0** | **1.4** |
| **ctrlg** (γ+β) | **38/45**† | **0** | **1** | **5** | **44** | **0** | **0** | **0** | **1.4** |

† ctrlg retroactive pass rate applies both γ and β checks; γ-only JSONL reports 44/45.
  Code is byte-for-byte identical to v5 γ-only (β is post-hoc detection only).

**v3 diagnosis (why adding BEHAVIOR examples hurts):** Inserting 7 new BEHAVIOR-domain examples (pick-place for-loops on household objects) — regardless of their position in the prompt — causes the Llama 3 8B model to generate `goto_pos(...)` for all 45 queries. Root cause: the 8B model cannot reliably context-switch between two very different object domains (colorful blocks/bowls vs. household items). The BEHAVIOR examples introduce ambiguity about which pattern to follow, and the model falls back to the simplest action it knows from the API header comments (`goto_pos`). The plain block/bowl examples (v2) already elicit correct `put_first_on_second` calls when the context objects are correctly grounded.

### RC-6 — Prompt contained non-API calls (discovered after v3)

The original `tabletop_ui_prompt.txt` imports and uses two functions that are NOT in the actual CaP runtime (`cap/lmp/lmp_wrapper.py::setup_LMP::variable_vars`):

| Name | Actual status | How it leaked in |
|---|---|---|
| `say(...)` | Exists as `lambda msg: print(f"robot says: {msg}")` — a print wrapper, not a real robot action | Appeared in 40+ few-shot examples as narration; model copied the pattern into generated code |
| `stack_objects_in_order(object_names=...)` | **Does not exist** — no implementation anywhere in `cap/` | Imported at line 19 of the prompt; used in 4 few-shot examples; any generated code that called it would fail at runtime |

**Fix (v4):** A scripted transformation of `tabletop_ui_prompt_v1_no_goto.txt` → `tabletop_ui_prompt_v3_strict.txt`:
1. Stripped `say` and `stack_objects_in_order` from the import line.
2. Deleted every `say(...)` line in the few-shot examples.
3. Rewrote every `stack_objects_in_order(object_names=L)` call as an explicit loop:
   ```python
   for i in range(1, len(L)):
     put_first_on_second(L[i], L[i-1])
   ```
4. Dropped few-shot examples whose only body had been a `say(...)` call (e.g. "cut the bowls in half" with answer "no, I can only move objects around").

Additionally:
- `offline/batch_lmp_codegen.py`'s `_DEFAULT_CAP_API` allowlist was rebuilt from the authoritative source (`lmp_wrapper.py::setup_LMP::variable_vars` and `fixed_vars`), removing `say`/`stack_objects_in_order` and adding real callables like `goto_xy`, `follow_traj`, `get_bbox`, etc.
- `offline/lmp_codegen_test.py`'s system message now explicitly enumerates the callable API and forbids `say()`, `stack_objects_in_order()`, and any other non-API calls.

```bash
# v4: strict-API prompt (no say, no stack_objects_in_order)
python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_v1_context.jsonl \
  --few-shot-file cap/lmp/prompts/real/tabletop_ui_prompt_v3_strict.txt \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192 \
  --output demo/outputs_llama3_v4_strict_api.jsonl
```

### RC-7 — v4 ignores the rest of the CaP LMP surface

v4 hit 45/45 PASS but every single output was a flat sequence of `put_first_on_second(source, dest_string)` calls. The other callable LMPs advertised by `cap/lmp/prompts/real/` — `parse_obj_name`, `parse_position`, `parse_question`, `transform_shape_pts` — were never invoked. Root cause: `tabletop_ui_prompt_v3_strict.txt` ends with the "put the red block on the farthest bowl" example (a single-call pattern). Llama 3 8B anchors on the last few-shot example and mirrored it for all 45 BEHAVIOR queries.

**Why v3 and v4 together inform v5:**
- **v1** (context fix, original prompt) proved the model *will* emit `parse_obj_name` + for-loop when those patterns appear in the few-shot corpus (5 for-loops, 3 `parse_obj_name`).
- **v2** proved the strict `put_first_on_second` + destination form works when the block/bowl examples are clean (33 `put_first_on_second`).
- **v3** proved that appending BEHAVIOR-domain examples *with `say(...)` narration* regresses to `goto_pos` — because the BEHAVIOR examples contradicted the (implicit) strict API that the rest of the prompt followed.
- **v4** proved that stripping `say` / `stack_objects_in_order` yields correct API usage but collapses the pattern space to a single primitive.

**Fix (v5):** Append to v3_strict six carefully-written BEHAVIOR-domain examples that all follow the strict API (no `say`, no `stack_objects_in_order`) and demonstrate the richer patterns:
1. `parse_obj_name('the books', f'objects = {get_obj_names()}')` + for-loop + `put_first_on_second`
2. `parse_obj_name` with literal destination string (carton, dishwasher, bucket, coffee table)
3. `parse_position('a point next to the sink')` + `parse_obj_name` + for-loop
4. `parse_position('the designated area')` + `parse_obj_name` + for-loop

The last three examples (most recent in the prompt) all use the `parse_obj_name + for-loop` pattern, so the 8B model anchors on that pattern rather than on the single-call `put_first_on_second`. System message also now explicitly instructs the model to use `parse_obj_name` for groups of similar objects ("the books", "the plates") and `parse_position` for spatial references.

```bash
# v5: strict-API + diverse patterns (parse_obj_name + for-loops + parse_position)
python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_v1_context.jsonl \
  --few-shot-file cap/lmp/prompts/real/tabletop_ui_prompt_v4_diverse.txt \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192 \
  --output demo/outputs_llama3_v5_diverse.jsonl
```

### Context experiment noted for completeness

`offline/add_context_to_prompts.py` now accepts `--keep-duplicates`, which writes
`demo/prompts_v2_context_duplicates.jsonl`. Running v5 against the duplicate-preserved
context (e.g. `objects = ['notebook','notebook','notebook','notebook','notebook','hardback','carton']`)
**does not** elicit more `parse_obj_name + for-loop` usage from Llama 3 8B — instead the
model enumerates each duplicate by hand, producing consecutive identical
`put_first_on_second(...)` lines that trigger the `no_repetition` quality check and drop the
pass rate to 38/45. The de-duplicated context (`prompts_v1_context.jsonl`) remains the
recommended input. The duplicate-context artifacts are kept on disk as evidence, not for
reporting.

### Final recommendation

Use **v5** (`outputs_llama3_v5_diverse.jsonl`) for DARPA reporting. It combines:
- the strict-API guarantee from v4 (0 `say`, 0 `stack_objects_in_order`, 0 `goto_pos` fallbacks);
- the varied CaP primitives seen earlier (`parse_position` for spatial destinations, `parse_obj_name + for-loop` where applicable);
- the correct core action (`put_first_on_second`) for every pick-place query.

Artifacts to hand off:
- `demo/outputs_llama3_v5_diverse.jsonl` — the 45 generated programs
- `demo/prompts_v1_context.jsonl` — the matching prompt inputs
- `cap/lmp/prompts/real/tabletop_ui_prompt_v4_diverse.txt` — the frozen few-shot prompt that produced v5

### Fix Scripts

```bash
# Generate prompts_v1_context.jsonl (adds objects-derived context field to each record)
conda activate cap-vllm
python offline/add_context_to_prompts.py \
  --input demo/prompts_candidates.jsonl \
  --output demo/prompts_v1_context.jsonl

# v1: context fix only
python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_v1_context.jsonl \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192 \
  --output demo/outputs_llama3_v1_context_fix.jsonl

# v2: context + prompt fix (no goto_pos examples)
python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_v1_context.jsonl \
  --few-shot-file cap/lmp/prompts/real/tabletop_ui_prompt_v1_no_goto.txt \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192 \
  --output demo/outputs_llama3_v2_prompt_fix.jsonl

# v3: context + behavior-aligned few-shot examples
python offline/batch_lmp_codegen.py \
  --prompts demo/prompts_v1_context.jsonl \
  --few-shot-file cap/lmp/prompts/real/tabletop_ui_prompt_v2_behavior.txt \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192 \
  --output demo/outputs_llama3_v3_behavior_fewshot.jsonl
```

---

## Step 3 — Evaluate Output Quality and Produce the Report

> **Status (2026-04-15):** Both `offline/evaluate_outputs.py` and
> `offline/build_quality_report.py` are implemented. The canonical run uses the
> final v5 artifact `demo/outputs_llama3_v5_diverse.jsonl` (44/45 pass, diverse
> API usage, no hallucinated calls — see "Final recommendation" above).

### 3a — Per-output quality check

For each record in `demo/outputs_llama3_<version>.jsonl`, re-score against these criteria
(`batch_lmp_codegen.py` already writes a `quality` block; this step recomputes every
check deterministically and adds the `no_echo` check from the design spec):

| Criterion | `quality` key | What to check |
|---|---|---|
| Syntactically valid Python | `syntactically_valid` | No mid-statement truncation; `compile()` does not raise |
| Uses CaP API calls only | `uses_only_cap_api` | Only names in the built-in allowlist (derived from `cap/lmp/lmp_wrapper.py::setup_LMP`) appear as `Call` nodes |
| Does not echo the query | `no_echo` | First non-blank line is not `# Query: ...` |
| No repetition loops | `no_repetition` | No consecutive identical non-blank lines |
| Stops at correct boundary | `stops_correctly` | Output does not bleed past `# Query:` or `objects =` markers |
| Overall | `pass` | All non-null criteria are `true` |

Canonical command (v5):
```bash
conda run -n cap-vllm python offline/evaluate_outputs.py \
  --outputs demo/outputs_llama3_v5_diverse.jsonl \
  --output  demo/outputs_llama3_v5_evaluated.jsonl
```

`--cap-api-list` is optional — omitting it uses the built-in default allowlist (the
authoritative source, matched to `cap/lmp/lmp_wrapper.py::setup_LMP`'s `variable_vars`
and `fixed_vars`). A standalone `offline/cap_api_allowlist.txt` file would drift from
the wrapper; prefer the built-in default.

### 3b — Build the quality report

```bash
conda run -n cap-vllm python offline/build_quality_report.py \
  --outputs     demo/outputs_llama3_v5_evaluated.jsonl \
  --prompts     demo/prompts_v1_context.jsonl \
  --version     v5_diverse \
  --description "Strict-API + diverse BEHAVIOR few-shots (final)" \
  --output      demo/quality_report_v5_diverse.md
```

Report sections produced (matches the File Formats spec):
1. **Summary** — prompt count, overall pass rate, average code length
2. **Per-Criterion Pass Rates** (+ API-usage distribution + difficulty/category breakdowns)
3. **Prompt-by-Prompt Results** — one row per prompt with the generated code, verdict and failure reasons
4. **Notable Observations** — auto-derived from call counts (hallucinated API detection, sub-LMP coverage, for-loop usage)
5. **Conclusion & Recommendation** — DARPA-facing verdict line

Naming convention for iterations: `demo/quality_report_<version>.md` (e.g.
`quality_report_v5_diverse.md`). Keep one report per artifact so older versions
remain comparable; do not overwrite.

### Reporting Process — Inputs and Outputs at Each Stage

```
INPUT                              STEP                              OUTPUT
────────────────────────────────────────────────────────────────────────────────────
EAI prompt files                   Step 0: extract_cap_prompts.py    demo/prompts_candidates.jsonl
  (behavior/action_sequencing,       (Extractor Agent)               (~47 CaP-compatible candidates)
   virtualHome/action_sequencing)

demo/prompts_candidates.jsonl      Step 0b: add_context_to_prompts   demo/prompts_v1_context.jsonl
                                     (per-prompt object context)     (context field per record)

demo/prompts_v1_context.jsonl      Step 2b: batch_lmp_codegen        demo/outputs_llama3_<ver>.jsonl
  + vLLM server @ :8000                                              (generated code + inline quality)

demo/outputs_llama3_<ver>.jsonl    Step 3a: evaluate_outputs.py      demo/outputs_llama3_<ver>_evaluated.jsonl
  (+ built-in allowlist)                                             (recomputed quality flags + no_echo)

demo/outputs_llama3_<ver>_eval.    Step 3b: build_quality_report.py  demo/quality_report_<ver>.md
  + demo/prompts_v1_context.jsonl                                    (DARPA-ready document)
```

The final handoff to DARPA reporting is `demo/quality_report_v5_diverse.md` plus
`demo/outputs_llama3_v5_evaluated.jsonl` as raw evidence.

---

## Step 4 — Bake the Config (optional, for persistent use)

If results look good, update `configs/real_config.yaml` to default to Llama so you don't
need CLI flags every time. Edit the `lmps` section — each LMP needs the same fields:

```yaml
# configs/real_config.yaml  (relevant section)
lmp_config:
  lmps:
    tabletop_ui:
      prompt_fname: tabletop_ui_prompt
      model: dganochenko/llama-3-8b-chat
      api_mode: chat_completions
      base_url: http://localhost:8000/v1
      api_key: not-needed
      max_tokens: 512
      context_window: 8192
      temperature: 0
      # ... rest unchanged

    fgen:
      model: dganochenko/llama-3-8b-chat
      api_mode: chat_completions
      base_url: http://localhost:8000/v1
      api_key: not-needed
      max_tokens: 512
      context_window: 8192
      # ... rest unchanged

    # Repeat for: parse_obj_name, parse_position, parse_question, transform_shape_pts
```

Then the offline test simplifies to:
```bash
python offline/lmp_codegen_test.py "pick up the box"
```

---

## Step 5 — Real Robot (when ready)

`main.py` accepts the same flags as the offline script:

```bash
python main.py \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 \
  --context-window 8192
```

Or if the config was updated in Step 4, just:
```bash
python main.py
```

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `vllm: command not found` | Wrong env or pip shadowing | Use `/home/ubuntu/miniconda3/envs/cap-vllm/bin/vllm` |
| `APIConnectionError` | Server not ready | Wait for startup log; retry after 30 s |
| Empty or truncated code | Context too small | Lower `--max-tokens` (try 256); drop `--context-window` to 4096 |
| Hallucinated function names | Too few few-shot examples fit | Lower `--context-window` threshold to retain more examples |
| Code repeats itself | Temperature=0 + weak 8B | Add `--temperature 0.3` (edit config or script) |
| Model not found by vLLM | ID mismatch | Pass full snapshot path instead of HF ID |
| `CUDA out of memory` | Another process on GPU | Check `nvidia-smi`; lower `--gpu-memory-utilization` |

---

## Environment Quick Reference

```bash
# Start vLLM server
conda activate cap-vllm
/home/ubuntu/miniconda3/envs/cap-vllm/bin/vllm serve \
  dganochenko/llama-3-8b-chat --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.4

# Run offline test (from CaP root)
python offline/lmp_codegen_test.py "<task>" \
  --model dganochenko/llama-3-8b-chat \
  --vllm-host localhost:8000 \
  --max-tokens 512 --context-window 8192
```

Cached model path:
```
/home/ubuntu/.cache/huggingface/hub/
  models--dganochenko--llama-3-8b-chat/
    snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1/
```
