#!/usr/bin/env python3
"""
lmp_codegen_test.py — Offline CaP (Code-as-Policies) code generation test.

Purpose:
    Drive the CaP tabletop_ui LMP (Language-Model Program) in "generation-only" mode:
    build a prompt from few-shot examples + a natural-language query, call the LLM
    (OpenAI-compatible API or a local vLLM server), and print / save the generated
    Python robot-arm code — WITHOUT executing it in a simulator.

Typical usage:
    # Against a local vLLM server (Llama-3 8B):
    uv run python offline/lmp_codegen_test.py "pick up the box and place it on the battery" \
        --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --vllm-host localhost:8000 \
        --max-tokens 512 --context-window 8192 --show-prompt

    # Against OpenAI (requires OPENAI_API_KEY):
    uv run python offline/lmp_codegen_test.py "stack the tin foil on the box"

Key design decisions:
    - Uses LMP.build_prompt() to assemble the few-shot prompt; never executes generated code.
    - Supports both chat-completions API (local models, GPT-4-class) and the newer
      responses API (o-series reasoning models).
    - Markdown code-fence stripping and stop-token trimming ensure clean Python output.
    - Retries automatically on transient OpenAI API errors (rate-limit, connection, etc.).
"""

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from time import sleep
import sys

import numpy as np
from openai import APIConnectionError, APIError, RateLimitError

# ---------------------------------------------------------------------------
# Path setup: make the CaP package (cap/) importable when running this script
# directly from the offline/ subdirectory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cap.lmp.lmp import LMP, LMPFGen, _strip_echoed_query, _trim_at_stop_tokens
from cap.lmp.utils import load_config


def build_tabletop_lmp(config_path: str, few_shot_override: str | None = None,
                       model: str | None = None, vllm_host: str | None = None,
                       max_tokens: int | None = None, context_window: int | None = None) -> LMP:
    """
    Construct and return a tabletop_ui LMP ready for offline code generation.

    Args:
        config_path:        Path to the LMP config YAML (e.g. configs/real_config.yaml).
        few_shot_override:  Raw text of an alternative few-shot prompt file; if provided,
                            replaces the default tabletop prompt stored in the config.
        model:              LLM model name to use.  A '/' in the name signals a local
                            model that must be served via vLLM (e.g. 'Owner/ModelName').
        vllm_host:          host:port of the running vLLM server (required when model is local).
        max_tokens:         Override max output tokens in the LMP config.
        context_window:     Override context-window size; enables auto few-shot truncation.

    Returns:
        An LMP instance configured for generation-only (debug_mode=True prevents execution).
    """
    # Load the full YAML config; lmps_cfg is a dict keyed by LMP name.
    config = load_config(config_path)
    lmps_cfg = config["lmp_config"]["lmps"]

    # --- Optional per-LMP overrides ---
    if model is not None:
        # A '/' in the model name means it's a local HF-style checkpoint served by vLLM.
        is_local = "/" in model
        if is_local and not vllm_host:
            raise ValueError(f"Model '{model}' looks like a local model (contains '/') but no vllm_host was provided.")

        for lmp_cfg in lmps_cfg.values():
            lmp_cfg["model"] = model
            if is_local:
                # Point every LMP at the local vLLM OpenAI-compatible endpoint.
                lmp_cfg["base_url"] = f"http://{vllm_host}/v1"
                lmp_cfg["api_key"] = "not-needed"   # vLLM ignores auth by default
                lmp_cfg["api_mode"] = "chat_completions"
            if max_tokens is not None:
                lmp_cfg["max_tokens"] = max_tokens
            if context_window is not None:
                lmp_cfg["context_window"] = context_window

    # fixed_vars are always available inside generated code (e.g. np for array math).
    fixed_vars = {"np": np}
    variable_vars = {}  # populated at runtime by the robot environment; empty here

    # fgen handles sub-function generation (helper LMPs called from tabletop_ui code).
    fgen = LMPFGen(lmps_cfg["fgen"], fixed_vars, variable_vars)

    # Enable debug_mode so the LMP builds the prompt and tracks state but never exec()s.
    tabletop_cfg = copy.deepcopy(lmps_cfg["tabletop_ui"])
    tabletop_cfg["debug_mode"] = True

    lmp = LMP(
        "tabletop_ui",
        tabletop_cfg,
        fgen,
        fixed_vars,
        variable_vars,
    )

    # Optionally swap in a custom few-shot prompt (e.g. a domain-specific example file).
    if few_shot_override is not None:
        lmp._base_prompt = few_shot_override.strip()

    return lmp


def generate_code(lmp: LMP, query: str, context: str = "") -> dict[str, str]:
    """
    Call the LLM with the assembled few-shot prompt and return the generated code.

    Args:
        lmp:     A tabletop_ui LMP built by build_tabletop_lmp().
        query:   Natural-language task description (e.g. "pick up the box").
        context: Extra context prepended to the query — typically a comma-separated
                 list of objects visible on the table.

    Returns:
        A dict with keys:
            "prompt"    — the full prompt string sent to the LLM
            "use_query" — the final query string after any LMP transformations
            "code"      — the clean Python code string produced by the model
    """
    # build_prompt() inserts context + query into the few-shot template and optionally
    # truncates examples to fit within context_window tokens.
    prompt, use_query = lmp.build_prompt(query, context=context)

    # System message instructs the model to output only runnable Python (no markdown fences,
    # no imports, no explanation) — keeps post-processing simple and reliable.
    # Also forbids `say(...)` and `stack_objects_in_order(...)` which are NOT part of the
    # real CaP runtime API (cap/lmp/lmp_wrapper.py), and encourages using the full API
    # surface (parse_obj_name, parse_position, parse_question, transform_shape_pts)
    # instead of converging on a single primitive.
    system_msg = (
        "You are a helpful assistant that pays attention to the user's instructions "
        "and writes good python code for operating a robot arm in a tabletop "
        "environment. Only output python code with no explanation (code comments "
        "are ok) or formatting, it should be ready to parse directly and ran. Do "
        "not repeat my code, just complete/continue from my code. Do not import any "
        "packages that aren't already there, you should never use the import keyword "
        "since all packages you need are already imported in the examples. "
        "IMPORTANT: only call functions that the robot runtime actually supports — "
        "i.e. put_first_on_second, pick_place, goto_pos, move_up, move_down, "
        "open_gripper, close_gripper, get_obj_pos, get_obj_names, is_obj_visible, "
        "and the helper LMPs parse_obj_name, parse_position, parse_question, "
        "transform_shape_pts, plus basic Python. "
        "Do NOT call say(), stack_objects_in_order(), or any function not listed above. "
        "Use parse_obj_name(description, f'objects = {get_obj_names()}') to resolve "
        "groups of similar objects (e.g. 'the books', 'the plates', 'the pops') and "
        "iterate with a for-loop when the task involves multiple objects. Use "
        "parse_position(description) for spatial references like 'next to the sink' "
        "or 'the designated area'. Every line of code must be an executable robot action."
    )

    # Retry loop: transparently handles transient API errors (rate limits, timeouts).
    while True:
        try:
            if lmp._use_chat:
                # Standard chat-completions path — used by GPT-4-class and local vLLM models.
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
                # Responses API path — used by o-series reasoning models (o3, o4-mini, etc.)
                # which expose a different endpoint with a `reasoning` budget parameter.
                response = lmp._client.responses.create(
                    model=lmp._cfg.get("model", "gpt-5-nano"),
                    instructions=system_msg,
                    input=prompt,
                    reasoning={"effort": "low"},
                    max_output_tokens=lmp._cfg["max_tokens"],
                )
                raw_text = response.output_text or ""

            # --- Post-processing: strip any markdown code fences the model may have added ---
            stripped = raw_text.strip()
            if stripped.startswith("```python"):
                stripped = stripped[len("```python") :].strip()
            if stripped.startswith("```"):
                stripped = stripped[len("```") :].strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()

            # Remove any accidental echo of the query line that some models prepend.
            stripped = _strip_echoed_query(stripped, use_query)
            # Cut off at CaP stop tokens (e.g. "objects = ") to prevent the model from
            # generating the next example's context header.
            code_str = _trim_at_stop_tokens(stripped, lmp._stop_tokens)
            break  # success — exit retry loop

        except (RateLimitError, APIConnectionError, APIError) as e:
            print(f"OpenAI API got err {e}")
            print("Retrying after 10s.")
            sleep(10)

    return {"prompt": prompt, "use_query": use_query, "code": code_str}


def _slugify(text: str, max_words: int = 10) -> str:
    """
    Convert an arbitrary string into a filesystem-safe lowercase slug.

    Rules:
      - Lowercase everything.
      - Replace any run of non-alphanumeric characters with a single underscore.
      - Strip leading/trailing underscores.
      - Truncate to at most `max_words` underscore-separated tokens so filenames
        stay readable (e.g. a 20-word query won't create a 200-char filename).

    Examples:
      "pick up the box and place it on the battery"
        → "pick_up_the_box_and_place_it_on_the_battery"
      "unsloth/Meta-Llama-3.1-8B-Instruct"
        → "unsloth_meta_llama_3_1_8b_instruct"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    tokens = slug.split("_")
    return "_".join(tokens[:max_words])


def _model_slug(model: str | None) -> str:
    """
    Derive a compact, readable slug from a model identifier.

    Steps:
      1. For HF-style "Owner/Model" paths, drop the owner prefix.
      2. Strip uninformative suffixes (-Instruct, -Chat, -GGUF, -it, -v<N>).
      3. Slugify and keep at most 5 tokens — preserving family, version, and size.

    Examples:
      "unsloth/Meta-Llama-3.1-8B-Instruct" → "meta_llama_3_1_8b"
      "meta-llama/Meta-Llama-3-8B-Instruct" → "meta_llama_3_8b"
      "gpt-5-nano"                           → "gpt_5_nano"
    Falls back to "default_model" when model is None.
    """
    if model is None:
        return "default_model"
    # Drop owner prefix (everything before the last '/').
    name = model.split("/")[-1]
    # Strip common uninformative suffixes (case-insensitive).
    name = re.sub(r"[-_](instruct|chat|gguf|it|hf|v\d+)$", "", name, flags=re.IGNORECASE)
    return _slugify(name, max_words=5)


def save_result(
    result: dict[str, str],
    query: str,
    model: str | None,
    context: str,
    save_dir: Path,
    include_prompt: bool = False,
) -> Path:
    """
    Save the generation result as a Markdown file under save_dir.

    Filename convention (mirrors llama3_local_vllm_plan.md):
        {model_slug}_{query_slug}.md

    Example:
        model  = "unsloth/Meta-Llama-3.1-8B-Instruct"
        query  = "pick up the box and place it on the battery"
        → meta_llama_3_1_8b_instruct_pick_up_the_box_and_place_it_on_the_battery.md

    File layout:
        # <query>
        metadata table (model, context, timestamp)
        ## Generated Code  — fenced Python block
        ## Prompt          — fenced text block (only when include_prompt=True)

    Args:
        result:         Dict returned by generate_code() (keys: prompt, use_query, code).
        query:          Original natural-language query string.
        model:          Model identifier passed on the CLI (None = default OpenAI model).
        context:        Object-list context string used during generation.
        save_dir:       Directory to write the file into (created if absent).
        include_prompt: Whether to append the full prompt to the saved file.

    Returns:
        Path of the written file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: {model_slug}_{first-4-words-of-query}.md
    # e.g. meta_llama_3_1_8b_pick_up_the_box.md
    filename = f"{_model_slug(model)}_{_slugify(query, max_words=4)}.md"
    out_path = save_dir / filename

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_display = model if model is not None else "default (gpt-5-nano)"

    lines = [
        f"# {query}",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **Model** | `{model_display}` |",
        f"| **Context (objects)** | {context} |",
        f"| **Generated at** | {timestamp} |",
        "",
        "## Generated Code",
        "",
        "```python",
        result["code"],
        "```",
        "",
    ]

    if include_prompt:
        lines += [
            "## Full Prompt",
            "",
            "```",
            result["prompt"],
            "```",
            "",
        ]

    out_path.write_text("\n".join(lines))
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run CaP tabletop_ui LMP in generation-only mode.",
    )
    # Positional: the natural-language task the robot should perform.
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
    # Model selection: pass 'Owner/Model' (with '/') for a local vLLM-served checkpoint.
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
        "--save",
        action="store_true",
        help="Save the result as a Markdown file under --save-dir.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(REPO_ROOT / "demo"),
        help="Directory for saved Markdown files (default: <repo_root>/demo/).",
    )
    args = parser.parse_args()

    # Load custom few-shot examples from file if provided.
    few_shot_override = None
    if args.few_shot_file:
        few_shot_override = Path(args.few_shot_file).read_text()

    lmp = build_tabletop_lmp(args.config, few_shot_override=few_shot_override,
                             model=args.model, vllm_host=args.vllm_host,
                             max_tokens=args.max_tokens, context_window=args.context_window)
    result = generate_code(lmp, args.query, context=args.context)

    # --- Output: JSON mode (machine-readable) or human-readable text ---
    if args.json:
        print(
            json.dumps(
                {
                    "query": result["use_query"],
                    "code": result["code"],
                    # Only include prompt in JSON output when --show-prompt is also set.
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

    # Optionally persist the result to a Markdown file.
    if args.save:
        saved_path = save_result(
            result,
            query=args.query,
            model=args.model,
            context=args.context,
            save_dir=Path(args.save_dir),
            include_prompt=args.show_prompt,
        )
        print(f"\nSaved → {saved_path}")


if __name__ == "__main__":
    main()
