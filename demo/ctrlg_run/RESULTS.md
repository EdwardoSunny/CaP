# CaP Two-Level Ctrl-G Results

**Model**: `dganochenko/llama-3-8b-chat` via vLLM  
**Prompts**: 45 BEHAVIOR AS tasks (all CaP-compatible from 100 EAI prompts)  
**Few-shot**: `tabletop_ui_prompt_v4_diverse.txt`  
**Date**: 2026-04-16

## Coverage

100 EAI BEHAVIOR AS prompts were screened. 55 were **blocked** by state-change
predicates (cleaning, cooking, washing, etc.) incompatible with CaP's
pick-and-place API. The remaining 45 form the full CaP-compatible set.

## Summary

| Mode | PASS | FAIL | Rate | What it does |
|------|------|------|------|-------------|
| **baseline** | 35 | 10 | 78% | No constraints (raw LLM) |
| **gamma (γ)** | 38 | 7 | 84% | logit_bias bans + stop strings |
| **ctrlg (γ+β)** | 38 | 7 | 84% | γ decoding + β post-hoc rejection |

**Note on PASS/FAIL accounting**: The γ-only JSONL records `overall_pass`
using only γ checks (44 PASS, 1 FAIL). The table above retroactively applies
**both γ and β** to all three modes for fair comparison. This means 6 prompts
(c001, c010, c013, c021, c022, c023) pass the γ-only JSONL check but fail
the unified γ+β check.

**γ constraint value**: +6 pp (78% → 84%). γ eliminates `goto_pos` calls
that the baseline generates for 5 prompts (c001, c031, c032, c037, c038).
However, c015 is a regression: baseline passes but γ/ctrlg truncate the
output mid-f-string, resulting in a net +3 prompt swing.

**β adds detection, not correction**: β flags violations (identity_action,
offscene_destination) but does not change generation — it is post-hoc only.
The generated code between γ-only and ctrlg runs is **byte-for-byte
identical** across all 45 prompts. Without β, the γ-only run would report
44/45 PASS (98%), hiding 6 real semantic problems that β catches.

**β uniquely reveals**: c001 and c010 would appear to PASS under γ-only
(valid syntax, no banned calls), but β flags their `offscene_destination`
violations. Similarly, c013/c021/c022/c023 pass γ but β catches the
`identity_action` error. These are genuine semantic bugs that would produce
nonsensical robot behavior at runtime.

## Failure Taxonomy

### 1. β:identity_action — 4 prompts (c013, c021, c022, c023)

All four share the same query: **"put all the cartons into the carton"**
with only one `carton` in `context_objects`. The model generates
`put_first_on_second('carton', 'carton')` — logically impossible (same source
and destination).

**Root cause**: The query itself is ambiguous. In the original BEHAVIOR
scene, there are multiple carton instances (`carton_66`, `carton_67`), but
the CaP context deduplicates to a single `carton`. The model has no way to
distinguish source from destination.

**Why c018 passes with the same query**: c018 also asks "put all the cartons
into the carton" but its `context_objects` include `['carton', 'notebook',
'hardback', 'coffee table', 'sweater']`. The model interprets it as "put the
books into the carton" and generates `put_first_on_second('notebook', 'carton')`
— avoiding the identity action. The richer context gives the model an
alternative interpretation.

**Fix options**:
- Preserve instance IDs in context: `objects = ['carton_66', 'carton_67']`
- Skip prompts with single-instance self-referential targets during extraction

### 2. β:offscene_destination — 2 prompts (c001, c010)

**Note**: c001 has **different failure modes** across runs:
- Baseline: `goto_pos(get_obj_pos('designated area'))` → fails on γ:no_goto_pos
- γ/ctrlg: `put_first_on_second('plywood', 'designated area')` → γ fixes
  goto_pos but exposes underlying β:offscene_destination

c010 fails on β:offscene_destination consistently across all three modes.

Queries: "move the plywoods/tiles **to the designated area**".
Context: `objects = ['plywood']` or `objects = ['tile']`.
Generated: `put_first_on_second('plywood', 'designated area')`.

**Root cause**: "designated area" is a concept in the query but not a real
object in the scene. The original BEHAVIOR task uses `room_floor_kitchen_0`
as the destination, but this was stripped during CaP extraction.

**Fix options**:
- Add the floor/room destination to `context_objects` during extraction
- Map "designated area" → `parse_position('the designated area')` in β check

### 3. γ:syntactically_valid — 1 prompt (c015)

Query: "put all the highlighters into the backpack" (7 context objects).
Generated: `highlighter_names = parse_obj_name('the highlighters', f'`
— truncated mid-f-string by 512-token limit.

**Root cause**: With 7 objects in context, the model generates a more complex
multi-step plan using `parse_obj_name` + for-loop. The f-string expression
is cut at `max_tokens=512`.

**Fix options**: Increase `max_tokens` to 768+ for prompts with >5 objects.

### 4. γ:no_goto_pos — 5 baseline failures (c001, c031, c032, c037, c038)

Baseline generates `goto_pos(...)` which is banned. γ logit_bias
successfully prevents this — 4 of these 5 (c031, c032, c037, c038) become
PASS under γ/ctrlg. c001 is also fixed for goto_pos but reveals an
underlying β:offscene_destination violation (see Section 2 above).

**This is the clearest win for γ constraints** — 4 prompts go from FAIL to
PASS purely from logit_bias banning `goto_pos` first-subword tokens.

## Output Files

| File | Mode | Description |
|------|------|-------------|
| `outputs_llama3_baseline.jsonl` | baseline | No constraints |
| `outputs_llama3_gamma.jsonl` | gamma | γ-only (logit_bias + stop) |
| `outputs_llama3_ctrlg.jsonl` | ctrlg | γ + β post-hoc |

## Per-Prompt Results

| ID | Query | Base | γ | γ+β | Failure |
|----|-------|------|---|-----|---------|
| c001 | move the plywoods to the designated area | FAIL | FAIL | FAIL | base: γ:no_goto_pos; γ/ctrlg: β:offscene_destination |
| c002 | put all the books inside the carton | PASS | PASS | PASS | |
| c003 | arrange the objects next to the sink | PASS | PASS | PASS | |
| c004 | clear the table — put everything in the bucket | PASS | PASS | PASS | |
| c005 | collect all the pops and put them in the bucket | PASS | PASS | PASS | |
| c006 | fill the basket with all the items on the table | PASS | PASS | PASS | |
| c007 | place all the misplaced items on the table | PASS | PASS | PASS | |
| c008 | fill the bottom cabinet with the objects on the table | PASS | PASS | PASS | |
| c009 | put all the plates into the dishwasher | PASS | PASS | PASS | |
| c010 | move the tiles to the designated area | FAIL | FAIL | FAIL | β:offscene_destination |
| c011 | arrange the objects next to the saw | PASS | PASS | PASS | |
| c012 | place the cartons on top of the shelf | PASS | PASS | PASS | |
| c013 | put all the cartons into the carton | FAIL | FAIL | FAIL | β:identity_action |
| c014 | put all the markers into the bottom cabinet no top | PASS | PASS | PASS | |
| c015 | put all the highlighters into the backpack | PASS | FAIL | FAIL | γ:syntactically_valid |
| c016 | put all the backpacks into the mouse | PASS | PASS | PASS | |
| c017 | put all the backpacks into the toothpaste | PASS | PASS | PASS | |
| c018 | put all the cartons into the carton | PASS | PASS | PASS | |
| c019 | put all the cars into the pencil box | PASS | PASS | PASS | |
| c020 | put all the backpacks into the sunglass | PASS | PASS | PASS | |
| c021 | put all the cartons into the carton | FAIL | FAIL | FAIL | β:identity_action |
| c022 | put all the cartons into the carton | FAIL | FAIL | FAIL | β:identity_action |
| c023 | put all the cartons into the carton | FAIL | FAIL | FAIL | β:identity_action |
| c024 | put all the ashcans into the pop | PASS | PASS | PASS | |
| c025 | arrange the objects next to the sink | PASS | PASS | PASS | |
| c026 | put all the playthings into the carton | PASS | PASS | PASS | |
| c027 | put all the pumpkins into the bottom cabinet | PASS | PASS | PASS | |
| c028 | fill the bottom cabinet with the objects on the table | PASS | PASS | PASS | |
| c029 | put all the pastas into the countertop | PASS | PASS | PASS | |
| c030 | put all the plates into the top cabinet | PASS | PASS | PASS | |
| c031 | place the christmas trees on the breakfast table | FAIL | PASS | PASS | baseline: γ:no_goto_pos |
| c032 | arrange the objects next to the table lamp | FAIL | PASS | PASS | baseline: γ:no_goto_pos |
| c033 | place the books on the breakfast table | PASS | PASS | PASS | |
| c034 | place the chickens on the table knife | PASS | PASS | PASS | |
| c035 | arrange the objects next to the sink | PASS | PASS | PASS | |
| c036 | place the cabinets on top of the tray | PASS | PASS | PASS | |
| c037 | place the hardbacks on the coffee table | FAIL | PASS | PASS | baseline: γ:no_goto_pos |
| c038 | place the candles on the coffee table | FAIL | PASS | PASS | baseline: γ:no_goto_pos |
| c039 | put all the oatmeals into the bottom cabinet | PASS | PASS | PASS | |
| c040 | put all the breads into the bottom cabinet no top | PASS | PASS | PASS | |
| c041 | arrange the objects next to the newspaper | PASS | PASS | PASS | |
| c042 | arrange the objects next to the sink | PASS | PASS | PASS | |
| c043 | put all the cereals into the bottom cabinet no top | PASS | PASS | PASS | |
| c044 | put all the plates into the trash can | PASS | PASS | PASS | |
| c045 | place the socks on top of the sofa | PASS | PASS | PASS | |
