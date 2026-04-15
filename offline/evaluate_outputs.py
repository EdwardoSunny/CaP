#!/usr/bin/env python3
"""
evaluate_outputs.py — Step 3a: re-score a generated-outputs JSONL against the
full CaP quality criteria and write an evaluated copy.

Re-uses the quality checks from batch_lmp_codegen.py and adds the `no_echo`
check that's only specified in docs/llama3_local_vllm_plan.md (Step 3a).

The input file is normally produced by batch_lmp_codegen.py and already has a
`quality` block — this script recomputes every check (so a manually edited
output file is rescored deterministically) and writes the result to a new file.

Usage:
    python offline/evaluate_outputs.py \
        --outputs demo/outputs_llama3_v5_diverse.jsonl \
        --cap-api-list offline/cap_api_allowlist.txt \
        --output  demo/outputs_llama3_v5_evaluated.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_DIR = Path(__file__).resolve().parent
if str(OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFLINE_DIR))

from batch_lmp_codegen import (  # noqa: E402
    _check_syntactically_valid,
    _check_uses_only_cap_api,
    _check_no_repetition,
    _check_stops_correctly,
    _load_allowlist,
)


def _check_no_echo(code: str) -> bool:
    """First non-blank line must not start with '# Query:'."""
    for line in code.splitlines():
        if line.strip():
            return not line.lstrip().startswith("# Query:")
    return True


def evaluate_record(code: str, allowlist: set[str] | None) -> dict:
    syn  = _check_syntactically_valid(code)
    api  = _check_uses_only_cap_api(code, allowlist)
    echo = _check_no_echo(code)
    rep  = _check_no_repetition(code)
    stop = _check_stops_correctly(code)

    checks = [syn, echo, rep, stop]
    if api is not None:
        checks.append(api)
    return {
        "syntactically_valid": syn,
        "uses_only_cap_api":   api,
        "no_echo":             echo,
        "no_repetition":       rep,
        "stops_correctly":     stop,
        "pass":                all(checks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True,
                    help="Input outputs_llama3*.jsonl from batch_lmp_codegen.py")
    ap.add_argument("--output", required=True,
                    help="Output evaluated JSONL path")
    ap.add_argument("--cap-api-list", default=None,
                    help="Optional allowlist file; defaults to built-in CaP API set")
    args = ap.parse_args()

    in_path  = Path(args.outputs)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"[error] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    allowlist = _load_allowlist(Path(args.cap_api_list) if args.cap_api_list else None)

    n_pass = n_fail = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_f:
        for line in in_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            code = rec.get("generated_code", "")
            quality = evaluate_record(code, allowlist)
            rec["quality"] = quality
            out_f.write(json.dumps(rec) + "\n")
            n_pass += int(quality["pass"])
            n_fail += int(not quality["pass"])

    print(f"Evaluated {n_pass + n_fail} records: pass={n_pass}, fail={n_fail}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
