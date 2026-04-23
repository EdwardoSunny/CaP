# LLM Robot Policies Baselines

## Code as Policies (CaP)

A CaP implementation that uses a **locally-served Llama 3 8B** (via vLLM) in place
of the original cloud-hosted OpenAI backend. An end-to-end pipeline drives the CaP
tabletop\_ui LMP on BEHAVIOR-derived prompts and produces a DARPA-ready quality report.

### Local Llama 3 8B Pipeline (new)

End-to-end flow — every step has a dedicated script under `offline/`:

```
EAI benchmark prompt files (BEHAVIOR action_sequencing)
        │
        │  Step 0 — extract_cap_prompts.py
        │     (filter to CaP-compatible predicates:
        │      ontop / inside / nextto / onfloor / under)
        ▼
demo/prompts_candidates.jsonl                          (45 candidates)
        │
        │  Step 0b — verify_prompts.py   [optional; needs vLLM up]
        │     (7-criterion validator aligned to the
        │      Molmo + SAM2 + GraspGen + XArm7 pipeline)
        ▼
demo/prompts.jsonl                                     (≥6/7 passing)
        │
        │  Step 2a — add_context_to_prompts.py
        │     (inject per-prompt `objects = [...]` context)
        ▼
demo/prompts_v1_context.jsonl                          (45 contextualised)
        │
        │  Step 2b — batch_lmp_codegen.py
        │     (Llama 3 8B via vLLM @ :8000 → CaP code
        │      + inline quality flags)
        ▼
demo/outputs_llama3_v5_diverse.jsonl                   (45 generations)
        │
        │  Step 3a — evaluate_outputs.py
        │     (recompute 5 criteria + no_echo check)
        ▼
demo/outputs_llama3_v5_evaluated.jsonl                 (final quality flags)
        │
        │  Step 3b — build_quality_report.py
        ▼
demo/quality_report_v5_diverse.md                      (DARPA-ready report)
```

---

### File Structure — new / modified artifacts

```
CaP/
├── README.md                               ← this file
├── docs/
│   └── llama3_local_vllm_plan.md           Full plan + RC-1..RC-7 fix log
│
├── offline/                                Offline generation + eval scripts
│   ├── README.md
│   ├── extract_cap_prompts.py              Step 0  — BEHAVIOR prompt extractor
│   ├── verify_prompts.py                   Step 0b — 7-criterion prompt verifier
│   ├── add_context_to_prompts.py           Step 2a — inject per-prompt objects=[]
│   ├── batch_lmp_codegen.py                Step 2b — batch CaP codegen over JSONL
│   ├── lmp_codegen_test.py                 One-shot codegen helper used by 2b
│   ├── evaluate_outputs.py                 Step 3a — rescore outputs (+no_echo)
│   └── build_quality_report.py             Step 3b — render DARPA markdown report
│
├── cap/lmp/prompts/real/                   tabletop_ui few-shot prompt iterations
│   ├── tabletop_ui_prompt.txt              (live prompt — symlink/equivalent to latest)
│   ├── tabletop_ui_prompt_v0_original.txt  293 lines; original Google CaP prompt
│   ├── tabletop_ui_prompt_v1_no_goto.txt   214 lines; trailing goto_pos stripped (RC-2)
│   ├── tabletop_ui_prompt_v2_behavior.txt  258 lines; + BEHAVIOR few-shots (had say() — RC-6)
│   ├── tabletop_ui_prompt_v3_strict.txt    170 lines; strict API — no say / stack_... (v4)
│   └── tabletop_ui_prompt_v4_diverse.txt   202 lines; + parse_obj_name/position anchors (v5)
│
└── demo/                                   Versioned artifacts
    ├── prompts_candidates.jsonl            45 BEHAVIOR-extracted prompts
    ├── prompts_v1_context.jsonl            + per-prompt `objects = [...]` context
    ├── prompts_v2_context_duplicates.jsonl Experiment: keep duplicate instances (worse)
    │
    ├── outputs_llama3_v0_simple_error.jsonl      43/45 pass; all one-line goto_pos
    ├── outputs_llama3_v1_context_fix.jsonl       39/45; RC-1 fix (context)
    ├── outputs_llama3_v2_prompt_fix.jsonl        44/45; RC-2 fix (strip goto_pos examples)
    ├── outputs_llama3_v3_behavior_fewshot.jsonl  45/45 but all goto_pos — regression
    ├── outputs_llama3_v4_strict_api.jsonl        45/45, strict API, single primitive only
    ├── outputs_llama3_v5_diverse.jsonl   ✅      45/45 + diverse API composition
    ├── outputs_llama3_v5_evaluated.jsonl ✅      Step 3a rescored (canonical evidence)
    │
    ├── quality_report_v5_diverse.md      ✅      DARPA-ready quality report (final)
    │
    └── llama_3_8b_{pick_up_the_box, push_the_tin_foil, stack_the_blocks}.md
                                             Early single-prompt qualitative logs
```

Files marked ✅ are the final DARPA-reporting handoff.

---

### Quick start

```bash
# 0. Activate the env (vLLM + CaP deps)
conda activate cap-vllm

# 1. Start local vLLM server (separate terminal)
vllm serve dganochenko/llama-3-8b-chat \
    --port 8000 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.4

# 2. Generate code for all 45 prompts
python offline/batch_lmp_codegen.py \
    --prompts demo/prompts_v1_context.jsonl \
    --model   dganochenko/llama-3-8b-chat \
    --vllm-host localhost:8000 \
    --max-tokens 512 --context-window 8192 \
    --output demo/outputs_llama3_v5_diverse.jsonl

# 3a. Rescore quality (includes no_echo check)
python offline/evaluate_outputs.py \
    --outputs demo/outputs_llama3_v5_diverse.jsonl \
    --output  demo/outputs_llama3_v5_evaluated.jsonl

# 3b. Build the quality report
python offline/build_quality_report.py \
    --outputs     demo/outputs_llama3_v5_evaluated.jsonl \
    --prompts     demo/prompts_v1_context.jsonl \
    --version     v5_diverse \
    --description "Strict-API + diverse BEHAVIOR few-shots (final)" \
    --output      demo/quality_report_v5_diverse.md
```

Full design, root-cause log (RC-1 … RC-7), and per-version diversity table
live in [`docs/llama3_local_vllm_plan.md`](docs/llama3_local_vllm_plan.md).

---

### TODOs (original)

- Implement perception models for `llm_wrapper.py`
- Test motion primitives with actual XArm7 environment
  - See if we can get it so that we don't record when we use the XArm7 environment
- Collect actual constraints/boundaries for tabletop policies (reachable spaces of XArm7) and record in `llm_wrapper.py`

## SayCan
