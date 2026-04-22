# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**uril-cap** — LLM-driven robotic manipulation using Code as Policies (CaP). An XArm7 robot executes natural-language tasks by generating Python code or JSON action plans via LLMs (OpenAI API or local vLLM), with perception (SAM2 + Molmo VLM), diffusion-based grasp generation (GraspGen), and motion control.

## Setup & Commands

```bash
# Install dependencies (Python >=3.10, uses UV)
uv sync

# Run robot control (requires XArm7 + RealSense cameras)
python main.py
python main.py --model Qwen/Qwen3.5-27B-FP8 --vllm-host host:8000 --plan-mode python

# Offline code generation test (no robot needed)
python offline/lmp_codegen_test.py "pick up the box" --context "box, cup, plate"
python offline/lmp_codegen_test.py "stack blocks" --plan-mode json --show-prompt --json

# Dispatch a pre-written JSON action plan through the full stack (skips LLM)
uv run python test_full_plan.py --plan test_plans/test_baseline_dishwasher.json
uv run python test_full_plan.py --plan test_plans/test_dfa_dishwasher.json

# Motion-only test of a single hardcoded trajectory (no perception/grasp)
uv run python test_open_action.py --plan test_plans/open_dishwasher.json

# GraspGen tests
cd GraspGen && pytest tests/
pytest GraspGen/tests/test_math_utils.py -k "test_name"  # single test

# Home the robot
python home.py
```

## Architecture

### Pipeline Flow
```
Natural language task → LMP (LLM code gen) → Perception (SAM2+Molmo) → Grasp (GraspGen) → Motion (XArm7)
```

### Core Modules (`cap/`)

- **`lmp/`** — LLM policy layer. `LMP` class calls OpenAI/vLLM APIs to generate executable Python or JSON plans. `LMPWrapper` injects all strategies and handles unit conversion (meters↔mm, radians↔degrees). Few-shot prompts live in `lmp/prompts/real/`. Primitives on `LMPWrapper`: `pick(pos)` (approach→open→descend→close→retreat), `pick_place(pick, place)` (calls `pick` then places), plus `goto_pos`/`open_gripper`/`close_gripper`/`move_relative`, etc.
- **`perception/`** — Abstract `PerceptionModule` → `SAMMolmoPerception` uses Molmo VLM for point prediction + SAM2 for mask generation across multiple cameras.
- **`grasp/`** — Abstract `GraspStrategy` → `GraspGenStrategy` (NVIDIA diffusion model) or `HardcodeStrategy`. Returns `GraspResult` with 4x4 homogeneous transforms in meters.
- **`motion/`** — Abstract `MotionController` → `XArmMotionController`. Coordinates in mm for position, degrees for orientation.
- **`pipeline.py`** — `GraspPipeline` orchestrates perception→grasp→visualization with swappable components.
- **`segment_pc.py`** — Multi-camera point cloud capture, Molmo VLM queries, SAM2 mask selection, robot-frame transforms.

### Robot Environment (`ril_env/`)
- `RealEnv` — Context manager for multi-camera management, replay buffers, video recording.
- `XArmController` — Low-level XArm7 control.
- `MultiRealsense` / `Realsense` — Camera wrappers with synchronized timestamps.

### External Modules
- **`GraspGen/`** — Git submodule. NVIDIA diffusion-based 6-DOF grasp generation.
- **`cap/segment/segment-anything2/`** — SAM2 submodule (UV workspace member `segment-anything`).

### Configuration
- `configs/real_config.yaml` — LMP configs for 7 modules (tabletop_ui, parse_obj_name, fgen, etc.). Default model: `gpt-5-nano`.
- `configs/action_map.yaml` — VirtualHome action→method mapping (WALK, GRAB, PUTON, etc.).
- `configs/hardcoded_traj/` — hand-written trajectory files for `HardcodedActionAdapter`. See section below.
- `cap/camera_exposure_config.py` — Per-camera exposure settings.

### Hardcoded trajectory system (`cap/lmp/waypoint_action_adapter.py`)

`HardcodedActionAdapter` subclasses `VirtualHomeActionAdapter`. On construction it scans a `traj_dir` for `*.txt` files; each filename stem becomes a method override via `setattr` (e.g. `put_in.txt` replaces `adapter.put_in`). Actions without a `.txt` fall through to the parent class (so real perception-driven `grab` still works).

Trajectory file format (one step per line, comments with `#`):
```
x, y, z, rx, ry, rz              # move to absolute pose (mm, deg), default 3s
x, y, z, rx, ry, rz, duration    # custom duration (seconds)
OPEN                             # open gripper
CLOSE                            # close gripper
WAIT 0.5                         # sleep
END                              # stop parsing (rest of file ignored)
```

Wiring: `setup_LMP(..., plan_mode="json", hardcoded_traj_dir="configs/hardcoded_traj")` activates it. The adapter + action_map are stored on the returned LMP's `_cfg` dict (`_json_adapter`, `_action_map`) — `test_full_plan.py` retrieves them to bypass the LLM and call `run_json_plan` directly. `exec_hardcoded_traj(env, path)` is the standalone executor used internally.

### Test entry points
- `test_plans/*.json` — pre-written VH-format JSON plans (e.g. `{"OPEN": ["dishwasher","20"], "GRAB": ["blue mug","62"], ...}`).
- `test_full_plan.py` — identical RealEnv + `setup_LMP` bootstrap to `main.py`, but skips the LLM and dispatches a plan from `--plan <path>` through `run_json_plan`.
- `test_open_action.py` — minimal motion-only harness: bypasses `LMPWrapper` via a `MotionEnv` shim (only `goto_pos`/`open_gripper`/`close_gripper`). Useful for isolating hardcoded trajectories without loading perception/grasp.

## Key Conventions

- **Unit boundaries**: Perception/grasp operate in meters; motion control operates in mm/degrees. `LMPWrapper` handles conversion.
- **Dependency injection**: All strategies (perception, grasp, motion, visualizer) are injected into Pipeline/Wrapper via constructor.
- **Abstract base classes**: `PerceptionModule`, `GraspStrategy`, `MotionController` define swappable interfaces.
- **Plan modes**: `--plan-mode python` generates Python DSL code; `--plan-mode json` generates VirtualHome-style JSON action sequences.
- **Local vLLM**: Use `--model Owner/Model --vllm-host host:port`. Models with `/` in name are treated as local. Auto few-shot truncation via `--context-window`.
- **Grasp viz**: Set `ENABLE_GRASP_VIZ=1` env var to enable point cloud/mask/grasp visualization.
- **Molmo endpoint**: Default `http://scai4.cs.ucla.edu:8001/v1` with model `allenai/Molmo2-8B` (hardcoded in `cap/segment_pc.py`). Override via `MOLMO_BASE_URL` / `MOLMO_MODEL` env vars.
- **VH JSON plan format**: Dict-of-action-keys with array values, duplicate keys allowed (`{"GRAB": ["obj","id"], "PUTIN": ["obj","id","target","tid"]}`). Parsed by `parse_vh_plan` in `cap/lmp/json_dispatcher.py` using `object_pairs_hook` to preserve key order and duplicates. NOT the demo_results list-of-`{action,object}` format — that needs conversion before dispatch.
