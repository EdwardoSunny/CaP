# Offline LMP Code Generation

Test LLM code/plan generation without a robot or cameras. Generates the output that CaP would execute for a given task, in either Python-DSL or VirtualHome-style JSON.

## Usage

Run from the repo root:

```bash
# Default (gpt-5-nano via OpenAI API) — Python DSL
python offline/lmp_codegen_test.py "pick up the box and place it on the battery"

# VirtualHome-style JSON action plan
python offline/lmp_codegen_test.py "pick up the bread and put it on the shelf" \
  --plan-mode json

# Local vLLM model (works in either mode)
python offline/lmp_codegen_test.py "stack the blocks" \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --vllm-host scai5.cs.ucla.edu:8000 \
  --max-tokens 2048 \
  --context-window 8192

# Override with a different OpenAI model (no context-window needed)
python offline/lmp_codegen_test.py "push the tin foil to the left" --model gpt-5-mini
```

## Model selection

- `--model` sets the LLM for code generation. If the name contains `/` it is treated as a local vLLM model and `--vllm-host` is required.
- `--vllm-host` is the `host:port` of your vLLM server (e.g. `scai5.cs.ucla.edu:8000`). The script connects to `http://<host:port>/v1`.

## Context window handling

Smaller models (e.g. Llama-3-8B with 8192 context) may not fit the full few-shot prompt. Use these flags together:

- `--max-tokens` caps the max output tokens (e.g. `2048`).
- `--context-window` tells the system the model's total context size in tokens (e.g. `8192`). When set, few-shot examples are **automatically dropped from the end** until the prompt fits. Instructions, imports, and the query are never truncated.

If `--context-window` is not set, no truncation happens (fine for large-context models like gpt-5-nano).

## Other options

| Flag | Description |
|------|-------------|
| `--context` | Comma-separated object list (default: `box, battery, plastic, tin foil`) |
| `--config` | Path to config YAML (default: `configs/real_config.yaml`) |
| `--few-shot-file` | Custom prompt file to replace the default few-shot examples |
| `--show-prompt` | Print the full prompt sent to the LLM |
| `--json` | Output results as JSON |
| `--plan-mode` | `python` (default) or `json` — selects the Code-as-Policies DSL or VirtualHome-style action plan. |

## Plan modes

- `--plan-mode python` loads the `tabletop_ui` LMP and its Python few-shot prompt.
- `--plan-mode json` loads the `tabletop_ui_json` LMP, uses a planner system prompt, and prints the raw JSON plan from the model **plus** a parsed view. The parser tolerates ``` ```json ``` fences, trailing prose, and VirtualHome duplicate keys (e.g. repeated `"WALK"`). No robot is touched — this script only generates text.
