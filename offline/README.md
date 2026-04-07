# Offline LMP Code Generation

Test LLM code generation without a robot or cameras. Generates the Python code that CaP would execute for a given task.

## Usage

Run from the repo root:

```bash
# Default (gpt-5-nano via OpenAI API)
python offline/lmp_codegen_test.py "pick up the box and place it on the battery"

# With a local vLLM model
python offline/lmp_codegen_test.py "stack the blocks" \
  --model Qwen/Qwen3.5-27B-FP8 \
  --vllm-host scai4.cs.ucla.edu:8000

# Override with a different OpenAI model
python offline/lmp_codegen_test.py "push the tin foil to the left" --model gpt-5-mini
```

## Model selection

- `--model` sets the LLM for code generation. If the name contains `/` it is treated as a local vLLM model and `--vllm-host` is required.
- `--vllm-host` is the `host:port` of your vLLM server (e.g. `scai4.cs.ucla.edu:8000`). The script connects to `http://<host:port>/v1`.

## Other options

| Flag | Description |
|------|-------------|
| `--context` | Comma-separated object list (default: `box, battery, plastic, tin foil`) |
| `--config` | Path to config YAML (default: `configs/real_config.yaml`) |
| `--few-shot-file` | Custom prompt file to replace the default few-shot examples |
| `--show-prompt` | Print the full prompt sent to the LLM |
| `--json` | Output results as JSON |
