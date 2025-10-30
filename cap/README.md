# Cap - Multi-Camera Point Cloud Capture Package

A Python package for capturing and merging point clouds from multiple RealSense cameras, with optional AI-powered object segmentation.

## Features

- **Multi-camera capture**: Merge point clouds from multiple RealSense cameras
- **Robot frame transformation**: Transform point clouds to robot coordinate frame using calibration
- **AI segmentation**: Optional SAM2 + CLIP based object segmentation
- **Flexible usage**: Use as standalone scripts or import as a package

## Package Structure

```
cap/
├── pc.py                      # Basic point cloud capture and merging
├── segment_pc.py              # Point cloud capture with segmentation
├── camera_exposure_config.py  # Camera exposure settings
└── README.md                  # This file
```

## Installation

The package expects the following directory structure:

```
project_root/
├── cap/                  # This package
├── transforms/
│   └── transforms.npy    # Camera calibration transforms
├── configs/
│   └── sam2.1/
│       └── *.yaml        # SAM2 model configs
└── ckpt/
    └── *.pt              # Model checkpoints
```

## Usage

### Method 1: As Standalone Scripts

Run directly from the `cap/` directory:

```bash
cd cap/

# Basic capture (no segmentation)
python pc.py

# Capture with segmentation
python segment_pc.py
```

### Method 2: Import as Package (Recommended)

Run from the project root directory:

```python
from pathlib import Path
from cap.pc import RobotFrameMerger

# Create merger
merger = RobotFrameMerger(
    camera_serials=["327122079374", "317422074281"],
    calib_file=Path("transforms/transforms.npy"),
    max_depth=2.0,
    min_depth=0.1,
)

# Capture point cloud
points, colors = merger.capture_merged_pointcloud()

# Save
merger.save_pointcloud(points, colors, filename="output")

# Cleanup
merger.cleanup()
```

### With Segmentation

```python
from pathlib import Path
from cap.segment_pc import RobotFrameMerger, load_segmentation_models
import torch

# Load AI models
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_gen, clip_model, clip_proc = load_segmentation_models(
    sam2_checkpoint=Path("ckpt/sam2.1_hiera_large.pt"),
    sam2_config=Path("configs/sam2.1/sam2.1_hiera_l.yaml"),
    clip_model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    device=device
)

# Create merger with segmentation
merger = RobotFrameMerger(
    camera_serials=["327122079374", "317422074281"],
    calib_file=Path("transforms/transforms.npy"),
    sam_generator=sam_gen,
    clip_model=clip_model,
    clip_processor=clip_proc,
    device=device,
)

# Capture with segmentation
points, colors = merger.capture_merged_pointcloud(
    text_prompt="cup"  # Specify object to segment
)

# Save and cleanup
merger.save_pointcloud(points, colors, filename="segmented_output")
merger.cleanup()
```

## Configuration

### Camera Exposure

Edit `camera_exposure_config.py`:

```python
DEPTH_EXPOSURE = 3000  # Microseconds, or None for auto
RGB_EXPOSURE = None    # Microseconds, or None for auto
```

### Camera Serials

Update the camera serial numbers in your code:

```python
camera_serials = ["327122079374", "317422074281"]
```

## API Reference

### `RobotFrameMerger` (pc.py)

**Constructor:**
```python
RobotFrameMerger(
    camera_serials: list[str],
    calib_file: str | Path,
    max_depth: float = 2.0,
    min_depth: float = 0.1
)
```

**Methods:**
- `capture_merged_pointcloud()` → `(points, colors)`: Capture and merge point clouds
- `save_pointcloud(points, colors, filename)`: Save point cloud to PLY and TXT
- `visualize_pointcloud(points, colors)`: Visualize with Open3D
- `cleanup()`: Stop all cameras

### `RobotFrameMerger` (segment_pc.py)

Same as above, plus:

**Constructor additional args:**
```python
RobotFrameMerger(
    ...,
    sam_generator=None,      # SAM2 mask generator
    clip_model=None,         # CLIP model
    clip_processor=None,     # CLIP processor
    device="cuda"            # Device for models
)
```

**Methods:**
- `capture_merged_pointcloud(text_prompt=None)`: Capture with optional segmentation

### `load_segmentation_models()`

```python
load_segmentation_models(
    sam2_checkpoint: str | Path,
    sam2_config: str | Path,
    clip_model_name: str,
    device: str = "cuda"
) → (sam_generator, clip_model, clip_processor)
```

## Examples

See `../test_cap_package.py` for complete working examples.

## Notes

- Paths should be provided relative to where you run your script from
- When using as a package, import from the project root directory
- Camera configuration is loaded from `camera_exposure_config.py`
- Calibration transforms must be pre-computed and saved in `transforms.npy`
