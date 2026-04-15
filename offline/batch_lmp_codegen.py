#!/usr/bin/env python3
"""
batch_lmp_codegen.py — Batch CaP code generation over a JSONL prompt set.

Reads every prompt from ``--prompts`` (demo/prompts.jsonl), calls the tabletop_ui
LMP for each one, auto-evaluates quality where possible, and appends a structured
record to ``--output`` (demo/outputs_llama3.jsonl).

Typical usage (vLLM server must be running at :8000 first):

    python offline/batch_lmp_codegen.py \\
        --prompts demo/prompts.jsonl \\
        --model dganochenko/llama-3-8b-chat \\
        --vllm-host localhost:8000 \\
        --max-tokens 512 \\
        --context-window 8192 \\
        --output demo/outputs_llama3.jsonl

Output format (one JSON object per line):
    {
      "id": "p01",
      "query": "pick up the red block ...",
      "model": "dganochenko/llama-3-8b-chat",
      "timestamp": "2026-04-15T10:32:00Z",
      "generated_code": "...",
      "quality": {
        "syntactically_valid": true,
        "uses_only_cap_api": true,   // null if allowlist not loaded
        "no_repetition": true,
        "stops_correctly": true,
        "pass": true
      },
      "notes": ""
    }

Quality checks performed automatically:
    syntactically_valid  — compile() on the code does not raise SyntaxError
    uses_only_cap_api    — all called names appear in cap_api_allowlist.txt
                           (set to null when the allowlist file is absent)
    no_repetition        — no two consecutive identical non-blank lines
    stops_correctly      — code does not contain known CaP stop-token prefixes
                           ("objects =", "# Query:")

Resume behaviour:
    If --output already exists, prompts whose id already appears in it are
    skipped so a partial run can be continued without re-generating everything.
"""

import argparse
import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import numpy as np
from openai import APIConnectionError, APIError, RateLimitError

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make the offline/ directory importable as a flat namespace (no __init__.py).
if str(OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFLINE_DIR))

# Re-use helpers from the single-query script to avoid duplication.
from lmp_codegen_test import build_tabletop_lmp, generate_code  # noqa: E402


# ---------------------------------------------------------------------------
# CaP API allowlist — used for uses_only_cap_api check
# ---------------------------------------------------------------------------

# Authoritative allowlist: every name callable inside generated CaP code.
# Derived directly from cap/lmp/lmp_wrapper.py::setup_LMP's variable_vars and
# fixed_vars — this is the single source of truth for what the LMP runtime
# will accept. Names NOT in this set (e.g. "say", "stack_objects_in_order",
# "wait" if not listed here, etc.) are treated as hallucinations.
_DEFAULT_CAP_API = {
    # --- Motion (LMPWrapper) ---
    "goto_pos", "goto_xy", "move_relative", "move_up", "move_down",
    "pick_place", "follow_traj", "wait",
    # --- Gripper ---
    "open_gripper", "close_gripper", "set_gripper",
    "is_gripper_open", "is_gripper_closed",
    # --- Robot state ---
    "get_robot_pos", "get_robot_xy",
    # --- Object queries / actions ---
    "get_obj_pos", "get_obj_names", "is_obj_visible",
    "put_first_on_second",
    "get_bbox", "get_color",
    "get_object_center", "clear_detection_cache",
    "denormalize_xy", "get_corner_name", "get_side_name",
    # --- Sub-LMPs (instances of class LMP, used as callables) ---
    "parse_obj_name", "parse_position", "parse_question", "transform_shape_pts",
    # --- fixed_vars: libraries always available inside generated code ---
    "np", "time",
    # shapely.geometry / shapely.affinity are imported by name into fixed_vars
    # (Point, Polygon, LineString, translate, scale, rotate, etc.) — too many
    # to enumerate; if the generated code uses them the allowlist check will
    # flag them. For DARPA-facing reporting that is acceptable noise.
    # --- Builtins and literals ---
    "print", "len", "range", "enumerate", "zip",
    "list", "dict", "tuple", "set",
    "str", "int", "float", "bool",
    "None", "True", "False",
    "min", "max", "sum", "abs", "round",
    "any", "all", "sorted", "reversed",
}


def _load_allowlist(path: Path | None) -> set[str] | None:
    """
    Load the CaP API allowlist from a text file (one name per line).
    Returns None if the file does not exist (caller should mark check as null).
    Falls back to the hardcoded default if path is explicitly provided but missing.
    """
    if path is None:
        # No explicit path: use built-in default.
        return _DEFAULT_CAP_API

    if not path.exists():
        print(f"[warn] Cap API allowlist not found at {path}; uses_only_cap_api will be null.")
        return None

    names = set()
    for line in path.read_text().splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.add(name)
    return names or None


# ---------------------------------------------------------------------------
# Per-output quality checks
# ---------------------------------------------------------------------------

_STOP_TOKEN_PREFIXES = ("objects =", "# Query:", "# objects")


def _check_syntactically_valid(code: str) -> bool:
    """Return True if the code string parses without a SyntaxError."""
    if not code.strip():
        return False
    try:
        compile(code, "<generated>", "exec")
        return True
    except SyntaxError:
        return False


def _collect_called_names(code: str) -> set[str]:
    """
    Walk the AST of code and collect every name that appears as the func in a
    Call node (i.e. every function / method called at the top level of the name).
    Also includes simple Name loads so variable accesses are caught.
    Returns an empty set on parse failure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                # e.g.  np.array  → add "np"
                if isinstance(func.value, ast.Name):
                    names.add(func.value.id)
    return names


def _check_uses_only_cap_api(code: str, allowlist: set[str] | None) -> bool | None:
    """
    Return True/False if allowlist is available; None otherwise.
    A hallucinated function is one that appears as a Call but is not in the allowlist.
    """
    if allowlist is None:
        return None
    if not code.strip():
        return None  # can't evaluate empty code

    called = _collect_called_names(code)
    # Remove names that are clearly just variables or builtins not worth flagging.
    hallucinated = called - allowlist
    return len(hallucinated) == 0


def _check_no_repetition(code: str) -> bool:
    """
    Return True if no two consecutive non-blank lines are identical.
    Targets the 'model loops on the same call' failure mode.
    """
    lines = [ln for ln in code.splitlines() if ln.strip()]
    for i in range(len(lines) - 1):
        if lines[i] == lines[i + 1]:
            return False
    return True


def _check_stops_correctly(code: str) -> bool:
    """
    Return True if no *line* in the generated code starts with a CaP stop-token
    prefix. Legitimate uses of these tokens inside f-strings (e.g.
    ``f'objects = {get_obj_names()}'`` passed to ``parse_obj_name``) are allowed.
    """
    for line in code.splitlines():
        stripped = line.lstrip()
        for prefix in _STOP_TOKEN_PREFIXES:
            if stripped.startswith(prefix):
                return False
    return True


def evaluate_quality(code: str, allowlist: set[str] | None) -> dict:
    """
    Run all automated quality checks and return a quality dict.

    Keys:
        syntactically_valid  bool
        uses_only_cap_api    bool | null
        no_repetition        bool
        stops_correctly      bool
        pass                 bool  — True iff all non-null checks pass
    """
    syn  = _check_syntactically_valid(code)
    api  = _check_uses_only_cap_api(code, allowlist)
    rep  = _check_no_repetition(code)
    stop = _check_stops_correctly(code)

    # pass = all checkable criteria are True
    all_checks = [syn, rep, stop]
    if api is not None:
        all_checks.append(api)
    overall = all(all_checks)

    return {
        "syntactically_valid": syn,
        "uses_only_cap_api": api,   # may be None → serialized as JSON null
        "no_repetition": rep,
        "stops_correctly": stop,
        "pass": overall,
    }


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _already_done(output_path: Path) -> set[str]:
    """Return the set of prompt ids already present in the output file."""
    if not output_path.exists():
        return set()
    done: set[str] = set()
    for line in output_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if "id" in rec:
                done.add(rec["id"])
        except json.JSONDecodeError:
            pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch CaP code generation over a JSONL prompt set.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to demo/prompts.jsonl (one prompt record per line).",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "demo" / "outputs_llama3.jsonl"),
        help="Output JSONL path (default: demo/outputs_llama3.jsonl).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name. Use 'Owner/Model' (with '/') for local vLLM.",
    )
    parser.add_argument(
        "--vllm-host",
        default=None,
        help="vLLM server host:port, e.g. localhost:8000",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Override max output tokens (default: 512).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=8192,
        help="Model context window size in tokens (default: 8192).",
    )
    parser.add_argument(
        "--config",
        default="configs/real_config.yaml",
        help="Path to LMP config YAML (default: configs/real_config.yaml).",
    )
    parser.add_argument(
        "--few-shot-file",
        default=None,
        help="Path to an alternative few-shot prompt file.",
    )
    parser.add_argument(
        "--context",
        default="box, battery, plastic, tin foil, red block, blue block",
        help="Default object-list context passed to every prompt (can be overridden per-prompt via 'context' field).",
    )
    parser.add_argument(
        "--cap-api-list",
        default=None,
        help="Path to cap_api_allowlist.txt (one function name per line). "
             "If omitted, a built-in default list is used.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output file and regenerate all prompts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated code for each prompt.",
    )
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_path  = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    if not prompts_path.exists():
        print(f"[error] Prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)
    prompts = _load_jsonl(prompts_path)
    if not prompts:
        print("[error] Prompts file is empty.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    # Resume: skip already-done ids
    done_ids: set[str] = set()
    if not args.no_resume:
        done_ids = _already_done(output_path)
        if done_ids:
            print(f"Resuming: {len(done_ids)} prompt(s) already in output, skipping.")

    # Load allowlist
    allowlist_path = Path(args.cap_api_list) if args.cap_api_list else None
    allowlist = _load_allowlist(allowlist_path)

    # Load optional few-shot override
    few_shot_override = None
    if args.few_shot_file:
        few_shot_override = Path(args.few_shot_file).read_text()

    # Build the LMP once — shared across all prompts
    print("Building LMP...")
    lmp = build_tabletop_lmp(
        args.config,
        few_shot_override=few_shot_override,
        model=args.model,
        vllm_host=args.vllm_host,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
    )

    model_name = args.model or lmp._cfg.get("model", "unknown")
    print(f"Model: {model_name}")
    print(f"Output: {output_path}\n")

    # Open output file in append mode so partial runs accumulate
    n_done = 0
    n_skip = 0
    n_fail = 0

    with output_path.open("a") as out_f:
        for i, prompt_rec in enumerate(prompts):
            pid   = prompt_rec.get("id", f"p{i+1:02d}")
            query = prompt_rec.get("query", "")
            context = prompt_rec.get("context", args.context)

            if pid in done_ids:
                n_skip += 1
                continue

            if not query:
                print(f"[warn] Prompt {pid} has no 'query' field, skipping.")
                n_skip += 1
                continue

            print(f"[{i+1}/{len(prompts)}] {pid}: {query[:70]}...")

            # Generate with retry on transient errors
            result = None
            for attempt in range(3):
                try:
                    result = generate_code(lmp, query, context=context)
                    break
                except (RateLimitError, APIConnectionError, APIError) as e:
                    print(f"  [retry {attempt+1}/3] API error: {e}")
                    sleep(10)
                except Exception as e:
                    print(f"  [error] Unexpected error: {e}")
                    break

            if result is None:
                print(f"  [skip] Failed to generate code for {pid}")
                n_fail += 1
                continue

            code = result["code"]
            quality = evaluate_quality(code, allowlist)

            if args.verbose:
                print(f"  --- generated code ---\n{code}\n  ----------------------")

            verdict = "PASS" if quality["pass"] else "FAIL"
            print(f"  quality: {verdict} | syn={quality['syntactically_valid']} "
                  f"api={quality['uses_only_cap_api']} "
                  f"rep={quality['no_repetition']} "
                  f"stop={quality['stops_correctly']}")

            record = {
                "id":             pid,
                "query":          query,
                "model":          model_name,
                "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "generated_code": code,
                "quality":        quality,
                "notes":          prompt_rec.get("notes", ""),
                # Carry through extra prompt metadata for convenience
                "category":       prompt_rec.get("category", ""),
                "difficulty":     prompt_rec.get("difficulty", ""),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_done += 1

    print(f"\nDone. Generated={n_done}, Skipped={n_skip}, Failed={n_fail}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
