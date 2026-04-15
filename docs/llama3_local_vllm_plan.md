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

## Diagnosis & Fix Log — Generated Code Quality Issues

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
| **v4** ✅ **BEST** | RC-6: remove non-API calls from few-shot prompt (`say`, `stack_objects_in_order`) and tighten system message | `tabletop_ui_prompt_v3_strict.txt` | `prompts_v1_context.jsonl` | `outputs_llama3_v4_strict_api.jsonl` | **45/45 pass**, **0 `say`**, **0 `stack_objects_in_order`**, **45/45 use `put_first_on_second`**, 0 `goto_pos` fallbacks, avg 1.5 lines |

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

**Recommendation:** Use **v4** (`outputs_llama3_v4_strict_api.jsonl`) for DARPA reporting — every generated line is an executable robot action with no narration stubs and no unimplemented calls.

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

### 3a — Per-output quality check

For each record in `demo/outputs_llama3.jsonl`, fill in the `quality` block using these criteria:

| Criterion | `quality` key | What to check |
|---|---|---|
| Syntactically valid Python | `syntactically_valid` | No mid-statement truncation; `compile()` does not raise |
| Uses CaP API calls only | `uses_only_cap_api` | `pick_and_place()`, `push()`, `goto_pos()`, etc. — no invented functions |
| Does not echo the query | `no_echo` | First line is not `# Query: ...` |
| No repetition loops | `no_repetition` | No consecutive identical calls |
| Stops at correct boundary | `stops_correctly` | Output does not bleed past `# Query:` or `objects =` markers |
| Overall | `pass` | All five criteria are `true` |

Automated check (where feasible):
```bash
python offline/evaluate_outputs.py \
  --outputs demo/outputs_llama3.jsonl \
  --cap-api-list offline/cap_api_allowlist.txt \
  --output demo/outputs_llama3_evaluated.jsonl
```

> **Note:** `offline/evaluate_outputs.py` and `offline/cap_api_allowlist.txt` do not yet exist.
> Until written, fill the `quality` block manually after inspecting each output.

### 3b — Build the quality report

Once `demo/outputs_llama3_evaluated.jsonl` is ready:

```bash
python offline/build_quality_report.py \
  --outputs demo/outputs_llama3_evaluated.jsonl \
  --prompts demo/prompts.jsonl \
  --output demo/quality_report.md
```

> **Note:** `offline/build_quality_report.py` does not yet exist. Until written, draft
> `demo/quality_report.md` manually using the structure described in the File Formats section.

### Reporting Process — Inputs and Outputs at Each Stage

```
INPUT                              STEP                              OUTPUT
────────────────────────────────────────────────────────────────────────────────────
EAI prompt files                   Step 0: extract_cap_prompts.py    demo/prompts_candidates.jsonl
  (behavior/action_sequencing,       (Extractor Agent)               (~47 CaP-compatible candidates)
   virtualHome/action_sequencing)

demo/prompts_candidates.jsonl      Step 0b: verify_prompts.py        demo/prompts.jsonl
  + vLLM server @ :8000              (Verifier Agent)                (10–20 validated, curated prompts)

demo/prompts.jsonl                 Step 2b: batch generation         demo/outputs_llama3.jsonl
  + vLLM server @ :8000

demo/outputs_llama3.jsonl          Step 3a: quality check            demo/outputs_llama3_evaluated.jsonl
  + cap_api_allowlist.txt                                            (quality flags per record)

demo/outputs_llama3_evaluated      Step 3b: report build             demo/quality_report.md
  + demo/prompts.jsonl                                               (DARPA-ready document)
```

The final handoff to DARPA reporting is `demo/quality_report.md` plus
`demo/outputs_llama3_evaluated.jsonl` as raw evidence.

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
