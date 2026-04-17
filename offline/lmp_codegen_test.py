#!/usr/bin/env python3
"""Generate CaP LMP code without executing generated code.

Supports two plan modes matching the online runtime:
  * --plan-mode python  (default) → Code-as-Policies Python DSL
  * --plan-mode json              → VirtualHome-style JSON action plans
"""

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

_PYTHON_SYSTEM_MSG = (
    "You are a helpful assistant that pays attention to the user's instructions "
    "and writes good python code for operating a robot arm in a tabletop "
    "environment. Only output python code with no explanation (code comments "
    "are ok) or formatting, it should be ready to parse directly and ran. Do "
    "not repeat my code, just complete/continue from my code. Do not import any "
    "packages that aren't already there, you should never use the import keyword "
    "since all packages you need are already imported in the examples."
)

_JSON_SYSTEM_MSG = (
    "You are a task planner for a robot. Output ONLY a single JSON object whose keys are "
    "action names (e.g. WALK, GRAB, PUTON, PUTIN, OPEN, CLOSE, SWITCHON, SWITCHOFF, LOOKAT, TOUCH) "
    "and whose values are argument arrays like [name, id] or [obj, obj_id, target, target_id]. "
    "Emit actions in the order they should execute. Do not explain, do not wrap in markdown, "
    "do not repeat the query."
)

_LMP_NAME_BY_MODE = {"python": "tabletop_ui", "json": "tabletop_ui_json"}


def build_tabletop_lmp(config_path: str, few_shot_override: str | None = None,
                       model: str | None = None, vllm_host: str | None = None,
                       max_tokens: int | None = None, context_window: int | None = None,
                       plan_mode: str = "python") -> LMP:
    if plan_mode not in _LMP_NAME_BY_MODE:
        raise ValueError(f"Unknown plan_mode: {plan_mode!r} (expected 'python' or 'json')")

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
            if max_tokens is not None:
                lmp_cfg["max_tokens"] = max_tokens
            if context_window is not None:
                lmp_cfg["context_window"] = context_window

    fixed_vars = {"np": np}
    variable_vars = {}

    fgen = LMPFGen(lmps_cfg["fgen"], fixed_vars, variable_vars)

    lmp_name = _LMP_NAME_BY_MODE[plan_mode]
    if lmp_name not in lmps_cfg:
        raise KeyError(
            f"Config {config_path} has no LMP entry '{lmp_name}' for plan_mode={plan_mode!r}"
        )
    tabletop_cfg = copy.deepcopy(lmps_cfg[lmp_name])
    tabletop_cfg["debug_mode"] = True

    lmp = LMP(
        lmp_name,
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
    output_format = lmp._cfg.get("output_format", "python")
    system_msg = _JSON_SYSTEM_MSG if output_format == "json_actions" else _PYTHON_SYSTEM_MSG

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

    parsed_plan = None
    if output_format == "json_actions":
        try:
            from cap.lmp.json_dispatcher import parse_vh_plan
            parsed_plan = [
                {"action": a, "args": list(args) if isinstance(args, (list, tuple)) else [args]}
                for a, args in parse_vh_plan(code_str)
            ]
        except Exception as e:
            parsed_plan = {"error": str(e)}

    return {
        "prompt": prompt,
        "use_query": use_query,
        "code": code_str,
        "output_format": output_format,
        "parsed_plan": parsed_plan,
    }


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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max output tokens (useful for smaller models).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Model context window size in tokens (enables auto few-shot truncation).",
    )
    parser.add_argument(
        "--plan-mode",
        choices=["python", "json"],
        default="python",
        help="'python' = Code-as-Policies DSL (default). 'json' = VirtualHome-style action plan.",
    )
    args = parser.parse_args()

    few_shot_override = None
    if args.few_shot_file:
        few_shot_override = Path(args.few_shot_file).read_text()

    lmp = build_tabletop_lmp(args.config, few_shot_override=few_shot_override,
                             model=args.model, vllm_host=args.vllm_host,
                             max_tokens=args.max_tokens, context_window=args.context_window,
                             plan_mode=args.plan_mode)
    result = generate_code(lmp, args.query, context=args.context)

    if args.json:
        print(
            json.dumps(
                {
                    "query": result["use_query"],
                    "output_format": result["output_format"],
                    "code": result["code"],
                    "parsed_plan": result["parsed_plan"],
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

    header = "GENERATED PLAN" if result["output_format"] == "json_actions" else "GENERATED CODE"
    print("=" * 80)
    print(header)
    print("=" * 80)
    print(result["code"])

    if result["parsed_plan"] is not None:
        print()
        print("=" * 80)
        print("PARSED ACTIONS")
        print("=" * 80)
        print(json.dumps(result["parsed_plan"], indent=2))


if __name__ == "__main__":
    main()
