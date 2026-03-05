#!/usr/bin/env python3
"""Generate CaP LMP code without executing generated code."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cap.lmp.lmp import LMP, LMPFGen
from cap.lmp.utils import load_config


def build_tabletop_lmp(config_path: str, few_shot_override: str | None = None) -> LMP:
    config = load_config(config_path)
    lmps_cfg = config["lmp_config"]["lmps"]

    fixed_vars = {"np": np}
    variable_vars = {}

    fgen = LMPFGen(lmps_cfg["fgen"], fixed_vars, variable_vars)

    tabletop_cfg = copy.deepcopy(lmps_cfg["tabletop_ui"])
    tabletop_cfg["debug_mode"] = True

    lmp = LMP(
        "tabletop_ui",
        tabletop_cfg,
        fgen,
        fixed_vars,
        variable_vars,
    )

    if few_shot_override is not None:
        lmp._base_prompt = few_shot_override.strip()

    return lmp


def main():
    parser = argparse.ArgumentParser(
        description="Run CaP tabletop_ui LMP in generation-only mode.",
    )
    parser.add_argument("query", help="Natural-language task prompt.")
    parser.add_argument(
        "--context",
        default="box, battery, plastic, tin foil",
        help="Optional extra context appended before the query (e.g. objects list).",
    )
    parser.add_argument(
        "--config",
        default="configs/real_config.yaml",
        help="Path to LMP config YAML.",
    )
    parser.add_argument(
        "--few-shot-file",
        default=None,
        help="Path to a prompt file to use as few-shot examples instead of the default tabletop prompt.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print full prompt sent to the LLM.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    args = parser.parse_args()

    few_shot_override = None
    if args.few_shot_file:
        few_shot_override = Path(args.few_shot_file).read_text()

    lmp = build_tabletop_lmp(args.config, few_shot_override=few_shot_override)
    result = lmp.generate_code(args.query, context=args.context)

    if args.json:
        print(
            json.dumps(
                {
                    "query": result["use_query"],
                    "code": result["code"],
                    "prompt": result["prompt"] if args.show_prompt else None,
                },
                indent=2,
            )
        )
        return

    if args.show_prompt:
        print("=" * 80)
        print("PROMPT")
        print("=" * 80)
        print(result["prompt"])
        print()

    print("=" * 80)
    print("GENERATED CODE")
    print("=" * 80)
    print(result["code"])


if __name__ == "__main__":
    main()
