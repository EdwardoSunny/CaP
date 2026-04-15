#!/usr/bin/env python3
"""
verify_prompts.py — Verifier Agent for CaP demo prompt set.

For each candidate in demo/prompts_candidates.jsonl, sends a structured validation
request to the local Llama 3 8B (via vLLM) and checks seven criteria derived from
the real XArm7 execution pipeline:

    Pipeline stage → Criterion
    ─────────────────────────────────────────────────────
    Molmo VLM           → molmo_visible      (object has color/shape descriptor)
    GraspGen point cloud → graspgen_compatible (object is 3D, convex, graspable)
    XArm7 single-arm    → single_arm         (no simultaneous multi-object ops)
    XArm7 workspace     → dest_reachable     (destination is on visible tabletop)
    LMP code execution  → cap_feasible       (task expressible with CaP API)
    LMP prompt→code     → unambiguous        (one interpretation given object list)
    Runtime safety      → no_contradiction   (no logical impossibilities)

Outputs:
    demo/prompts.jsonl         — candidates that pass --min-pass criteria (≥ 6/7 by default)
    demo/verify_report.jsonl   — full per-candidate verification results

Usage:
    # Requires vLLM server running:
    #   /home/ubuntu/miniconda3/envs/cap-vllm/bin/vllm serve dganochenko/llama-3-8b-chat \
    #       --port 8000 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.4

    python offline/verify_prompts.py \\
        --input demo/prompts_candidates.jsonl \\
        --output demo/prompts.jsonl \\
        --vllm-host localhost:8000 \\
        --model dganochenko/llama-3-8b-chat \\
        --min-pass 6 \\
        --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path
from time import sleep

from openai import OpenAI, APIConnectionError, APIError, RateLimitError

# ---------------------------------------------------------------------------
# CaP API reference (shown to the LLM in the verification prompt)
# ---------------------------------------------------------------------------

VERIFY_SYSTEM_PROMPT = """\
You are an expert robot task validator. The robot is a real XArm7 single-arm manipulator
with this execution pipeline:
  1. Molmo VLM identifies objects in the camera image using natural-language descriptions
  2. SAM2 segments the identified object and extracts its point cloud
  3. GraspGen generates a 6-DOF grasp pose from the point cloud
  4. XArm7 executes the pick-place trajectory

Your job is to evaluate whether a natural-language task instruction will work correctly
through all four stages of this real hardware pipeline. Be strict: if you have any doubt,
mark the criterion false."""

VERIFY_USER_TEMPLATE = """\
Robot API available to LLM-generated code:
  put_first_on_second, stack_objects_in_order, goto_pos, get_obj_pos, get_obj_names,
  parse_obj_name, parse_position, move_up, move_down, open_gripper, close_gripper,
  pick_place, is_obj_visible, say

Task instruction: "{query}"
Objects present on the table: {objects}

Evaluate against these 7 criteria. For each, answer true/false + one sentence:

1. molmo_visible: Does every object in the instruction have a color or shape descriptor
   that Molmo VLM can locate? (Fail if any object name is generic like "thing" or "object",
   or if two similar-looking objects appear without visual disambiguation.)
2. graspgen_compatible: Are all objects that need to be grasped 3D, convex, and graspable?
   (Fail if any object is paper, fabric, transparent, very flat, or smaller than ~2 cm.)
3. single_arm: Can the task be completed holding at most one object at a time?
   (Fail if the task implies simultaneously holding or moving two objects.)
4. dest_reachable: Is the destination a visible tabletop surface or another object on the table?
   (Fail if the destination is a drawer interior, the floor, off-table, or requires navigation.)
5. cap_feasible: Can the task be expressed entirely with the robot API listed above?
   (Fail if it requires cooking, cleaning, cutting, toggling switches, or object state changes.)
6. unambiguous: Is there exactly one reasonable interpretation given the object list?
   (Fail if pronouns or partial names could refer to multiple objects simultaneously.)
7. no_contradiction: Is the task free of logical impossibilities?
   (Fail if source = destination, if object count mismatches, or if the goal is already met.)

Respond ONLY with valid JSON (no markdown, no extra text):
{{"molmo_visible": true/false, "molmo_visible_reason": "...",
  "graspgen_compatible": true/false, "graspgen_compatible_reason": "...",
  "single_arm": true/false, "single_arm_reason": "...",
  "dest_reachable": true/false, "dest_reachable_reason": "...",
  "cap_feasible": true/false, "cap_feasible_reason": "...",
  "unambiguous": true/false, "unambiguous_reason": "...",
  "no_contradiction": true/false, "no_contradiction_reason": "...",
  "pass": true/false,
  "notes": "one-sentence overall assessment"}}"""


def build_client(vllm_host: str) -> OpenAI:
    return OpenAI(
        base_url=f"http://{vllm_host}/v1",
        api_key="not-needed",
    )


def verify_one(
    client: OpenAI,
    model: str,
    candidate: dict,
    min_pass: int = 4,
    max_retries: int = 3,
) -> dict:
    """
    Send one candidate to the LLM for verification. Returns the candidate dict
    augmented with 'verification' and 'verified_pass' fields.
    """
    objects_str = ", ".join(candidate.get("objects", []))
    user_msg = VERIFY_USER_TEMPLATE.format(
        query=candidate["query"],
        objects=objects_str or "(not specified — infer from task name)",
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            break
        except (RateLimitError, APIConnectionError, APIError) as e:
            if attempt < max_retries - 1:
                print(f"  [verify] API error: {e} — retrying in 5s")
                sleep(5)
            else:
                print(f"  [verify] API error after {max_retries} attempts: {e}")
                return {**candidate, "verification": None, "verified_pass": False}

    # --- Parse the JSON response ---
    # Strip any accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON object with regex
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                result = {}
        else:
            result = {}

    # Count criteria that passed
    criteria = [
        "molmo_visible", "graspgen_compatible", "single_arm", "dest_reachable",
        "cap_feasible", "unambiguous", "no_contradiction",
    ]
    n_pass = sum(1 for k in criteria if result.get(k) is True)
    verified_pass = n_pass >= min_pass

    # Override the LLM's 'pass' with our threshold-based decision
    result["pass"] = verified_pass
    result["criteria_passed"] = n_pass
    result["criteria_total"] = len(criteria)

    return {**candidate, "verification": result, "verified_pass": verified_pass}


def main():
    parser = argparse.ArgumentParser(
        description="Verify CaP demo prompt candidates using a local LLM as a judge."
    )
    parser.add_argument(
        "--input",
        default="demo/prompts_candidates.jsonl",
        help="Input candidates JSONL (from extract_cap_prompts.py or hand-authored).",
    )
    parser.add_argument(
        "--output",
        default="demo/prompts.jsonl",
        help="Output path for verified prompts (JSONL).",
    )
    parser.add_argument(
        "--report",
        default="demo/verify_report.jsonl",
        help="Output path for full per-candidate verification report (JSONL).",
    )
    parser.add_argument(
        "--vllm-host",
        default="localhost:8000",
        help="vLLM server host:port.",
    )
    parser.add_argument(
        "--model",
        default="dganochenko/llama-3-8b-chat",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--min-pass",
        type=int,
        default=6,
        help="Minimum number of criteria (out of 7) that must pass. Default: 6.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-candidate results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; just parse input and report what would be sent.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)

    if not in_path.exists():
        print(f"[verifier] ERROR: input file not found: {in_path}")
        sys.exit(1)

    candidates = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    print(f"[verifier] Loaded {len(candidates)} candidates from {in_path}")

    if args.dry_run:
        print("[verifier] Dry-run mode — showing first prompt that would be sent:")
        c = candidates[0]
        user_msg = VERIFY_USER_TEMPLATE.format(
            query=c["query"],
            objects=", ".join(c.get("objects", [])),
        )
        print("SYSTEM:", VERIFY_SYSTEM_PROMPT)
        print("USER:", user_msg)
        return

    client = build_client(args.vllm_host)

    # Verify connection
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if args.model not in available:
            print(f"[verifier] WARNING: model '{args.model}' not in available models: {available}")
        else:
            print(f"[verifier] Connected to vLLM — model '{args.model}' is available")
    except Exception as e:
        print(f"[verifier] ERROR: cannot connect to vLLM at {args.vllm_host}: {e}")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    passed = []
    all_results = []

    for i, candidate in enumerate(candidates):
        cid = candidate.get("id", f"c{i+1:03d}")
        query = candidate.get("query", "")
        print(f"[{i+1}/{len(candidates)}] {cid}: \"{query[:60]}\"", end=" ... ")

        result = verify_one(client, args.model, candidate, min_pass=args.min_pass)
        all_results.append(result)

        v = result.get("verification") or {}
        n_pass = v.get("criteria_passed", 0)
        verdict = "PASS" if result["verified_pass"] else "FAIL"
        print(f"{verdict} ({n_pass}/7)")

        if args.verbose and v:
            for k in ["molmo_visible", "graspgen_compatible", "single_arm", "dest_reachable",
                      "cap_feasible", "unambiguous", "no_contradiction"]:
                status = "✓" if v.get(k) else "✗"
                reason = v.get(f"{k}_reason", "")
                print(f"    {status} {k}: {reason}")
            if v.get("notes"):
                print(f"    notes: {v['notes']}")

        if result["verified_pass"]:
            # Add a clean 'quality' field for the output format
            passed.append({
                "id": cid,
                "query": query,
                "category": candidate.get("cap_operations", ["put_first_on_second"])[0]
                            .replace("put_first_on_second", "pick_place")
                            .replace("stack_objects_in_order", "multi_step"),
                "difficulty": candidate.get("difficulty", "medium"),
                "source": candidate.get("source", ""),
                "source_task": candidate.get("task_name", ""),
                "objects": candidate.get("objects", []),
                "cap_operations": candidate.get("cap_operations", []),
                "verification_summary": v.get("notes", ""),
            })

    # Write outputs
    with out_path.open("w") as f:
        for p in passed:
            f.write(json.dumps(p) + "\n")

    with report_path.open("w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"[verifier] Results: {len(passed)} passed / {len(candidates)} total")
    print(f"[verifier] Verified prompts → {out_path}")
    print(f"[verifier] Full report      → {report_path}")
    print()
    if len(passed) > 20:
        print(f"[verifier] NOTE: {len(passed)} prompts passed — "
              "manually trim to 10–20 diverse entries for the demo set.")
    elif len(passed) < 10:
        print(f"[verifier] NOTE: only {len(passed)} prompts passed — "
              "consider lowering --min-pass or hand-authoring additional prompts.")
    print()
    print("Next step: batch generation")
    print(f"  python offline/batch_lmp_codegen.py --prompts {out_path} "
          "--model dganochenko/llama-3-8b-chat --vllm-host localhost:8000 "
          "--output demo/outputs_llama3.jsonl")


if __name__ == "__main__":
    main()
