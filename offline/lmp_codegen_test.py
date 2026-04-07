#!/usr/bin/env python3
"""Generate CaP LMP code without executing generated code."""

import argparse
import copy
import json
from pathlib import Path
from time import sleep
import sys

import numpy as np
from openai import APIConnectionError, APIError, RateLimitError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cap.lmp.lmp import LMP, LMPFGen, _strip_echoed_query, _trim_at_stop_tokens
from cap.lmp.utils import load_config


def build_tabletop_lmp(config_path: str, few_shot_override: str | None = None,
                       model: str | None = None, vllm_host: str | None = None) -> LMP:
    config = load_config(config_path)
    lmps_cfg = config["lmp_config"]["lmps"]

    # Override model/host in all LMP configs
    if model is not None:
        is_local = "/" in model
        if is_local and not vllm_host:
            raise ValueError(f"Model '{model}' looks like a local model (contains '/') but no vllm_host was provided.")
        for lmp_cfg in lmps_cfg.values():
            lmp_cfg["model"] = model
            if is_local:
                lmp_cfg["base_url"] = f"http://{vllm_host}/v1"
                lmp_cfg["api_key"] = "not-needed"
                lmp_cfg["api_mode"] = "chat_completions"

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


def generate_code(lmp: LMP, query: str, context: str = "") -> dict[str, str]:
    prompt, use_query = lmp.build_prompt(query, context=context)
    system_msg = (
        "You are a helpful assistant that pays attention to the user's instructions "
        "and writes good python code for operating a robot arm in a tabletop "
        "environment. Only output python code with no explanation (code comments "
        "are ok) or formatting, it should be ready to parse directly and ran. Do "
        "not repeat my code, just complete/continue from my code. Do not import any "
        "packages that aren't already there, you should never use the import keyword "
        "since all packages you need are already imported in the examples."
    )

    while True:
        try:
            if lmp._use_chat:
                response = lmp._client.chat.completions.create(
                    model=lmp._cfg.get("model", "gpt-5-nano"),
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=lmp._cfg["max_tokens"],
                    temperature=lmp._cfg.get("temperature", 0),
                )
                raw_text = response.choices[0].message.content or ""
            else:
                response = lmp._client.responses.create(
                    model=lmp._cfg.get("model", "gpt-5-nano"),
                    instructions=system_msg,
                    input=prompt,
                    reasoning={"effort": "low"},
                    max_output_tokens=lmp._cfg["max_tokens"],
                )
                raw_text = response.output_text or ""

            stripped = raw_text.strip()
            if stripped.startswith("```python"):
                stripped = stripped[len("```python") :].strip()
            if stripped.startswith("```"):
                stripped = stripped[len("```") :].strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()

            stripped = _strip_echoed_query(stripped, use_query)
            code_str = _trim_at_stop_tokens(stripped, lmp._stop_tokens)
            break
        except (RateLimitError, APIConnectionError, APIError) as e:
            print(f"OpenAI API got err {e}")
            print("Retrying after 10s.")
            sleep(10)

    return {"prompt": prompt, "use_query": use_query, "code": code_str}


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
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name. Use 'Owner/Model' (with '/') for local vLLM.",
    )
    parser.add_argument(
        "--vllm-host",
        default=None,
        help="vLLM server host:port, e.g. scai4.cs.ucla.edu:8000",
    )
    args = parser.parse_args()

    few_shot_override = None
    if args.few_shot_file:
        few_shot_override = Path(args.few_shot_file).read_text()

    lmp = build_tabletop_lmp(args.config, few_shot_override=few_shot_override,
                             model=args.model, vllm_host=args.vllm_host)
    result = generate_code(lmp, args.query, context=args.context)

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
