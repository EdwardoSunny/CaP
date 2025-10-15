# Centralized Camera Configuration System

## Overview

All camera settings are now centralized in `configs/camera_config.yaml`. All scripts read from this single configuration file, making it easy to update camera parameters without modifying code.

## Files

### Configuration Files
- **`configs/camera_config.yaml`** - Main configuration file (YAML format)
- **`cap/camera_utils.py`** - Utility functions to load and apply configuration

### Updated Scripts
All scripts now use the centralized config:
- ✅ `cap/tune_pointcloud.py`
- ✅ `cap/tune_pointcloud_interactive.py`
- ✅ `cap/segment_pc.py`
- ✅ `cap/pc.py`
- ⚠️  `cap/debug_single_camera.py` (can be updated using same pattern)

## Configuration Structure

### Camera Settings
```yaml
camera_serials:
  - "317422074281"
  - "327122079374"

camera_settings:
  "317422074281":
    auto_exposure: 1
    gain: 64
    laser_power: 240
    exposure: null

  "327122079374":
    auto_exposure: 0
    exposure: 500
    gain: 16
    laser_power: 320
```

### Stream Configuration
```yaml
stream_config:
  width: 640
  height: 480
  fps: 30
  depth_format: "z16"
  color_format: "bgr8"
```

### Point Cloud Settings
```yaml
point_cloud:
  max_depth: 2.0
  min_depth: 0.1
  max_points: 100000
```

### Calibration Paths
```yaml
calibration:
  transforms_file: "./transforms/transforms.npy"
  icp_file: "./transforms/icp_tf.npy"
  manual_offset_file: "./transforms/manual_offset.npy"
  calib_units: "m"
  point_cloud_units: "m"
```

### Workspace Bounds
```yaml
workspace_bounds:
  enabled: false
  min_x: 0.15
  min_y: -0.4
  min_z: 0.08
  max_x: 1.0
  max_y: 0.35
  max_z: 0.6
```

### Segmentation Settings
```yaml
segmentation:
  sam2_checkpoint: "ckpt/sam2.1_hiera_large.pt"
  sam2_model_cfg: "configs/sam2.1/sam2.1_hiera_l.yaml"
  clip_model: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
  points_per_side: 32
  points_per_batch: 128
  pred_iou_thresh: 0.88
  stability_score_thresh: 0.95
  min_mask_region_area: 200.0
```

## Usage

### In Your Scripts

```python
from camera_utils import load_camera_config, configure_realsense_cameras

# Load configuration
config = load_camera_config()

# Configure cameras
configure_realsense_cameras(config)

# Access configuration values
camera_serials = config['camera_serials']
calib_config = config['calibration']
seg_config = config['segmentation']
```

### Helper Functions

```python
from camera_utils import (
    get_stream_config,
    get_point_cloud_config,
    get_calibration_config,
    get_workspace_bounds,
    get_segmentation_config
)

# Get specific config sections
stream_cfg = get_stream_config()
pc_cfg = get_point_cloud_config()
calib_cfg = get_calibration_config()
ws_bounds = get_workspace_bounds()  # Returns None if disabled
seg_cfg = get_segmentation_config()
```

## Benefits

### ✅ Single Source of Truth
- All camera settings in one place
- No need to hunt through multiple files
- Consistent settings across all scripts

### ✅ Easy Modification
- Change camera parameters without touching code
- Update YAML file and all scripts pick up changes
- Version control friendly

### ✅ Documented Settings
- YAML format is human-readable
- Comments explain each parameter
- Clear structure

### ✅ Flexible Configuration
- Enable/disable features (like workspace cropping)
- Switch between different segmentation prompts
- Adjust camera parameters per camera

## How to Modify Settings

### Example: Change Camera Exposure

Edit `configs/camera_config.yaml`:
```yaml
camera_settings:
  "327122079374":
    auto_exposure: 0
    exposure: 600  # Changed from 500
    gain: 16
    laser_power: 320
```

All scripts will automatically use the new exposure value!

### Example: Change SAM2 Model Settings

Edit `configs/camera_config.yaml`:
```yaml
segmentation:
```

Run `segment_pc.py` and it will segment the new object!

### Example: Enable Workspace Cropping

Edit `configs/camera_config.yaml`:
```yaml
workspace_bounds:
  enabled: true  # Changed from false
  min_x: 0.2
  min_y: -0.3
  # ... adjust bounds as needed
```

## Migration Pattern

If you need to update `debug_single_camera.py` or other scripts:

1. **Add import**:
   ```python
   from camera_utils import load_camera_config, configure_realsense_cameras
   ```

2. **Remove old config code**: Delete hardcoded CAMERA_SERIALS and configure_realsense_cameras() function

3. **Load config in main()**:
   ```python
   def main():
       config = load_camera_config()
       camera_serials = config['camera_serials']
       configure_realsense_cameras(config)
       # ... rest of code
   ```

4. **Use config values** instead of hardcoded constants

## Testing

Test the configuration system:
```bash
source .venv/bin/activate
python3 cap/camera_utils.py
```

Should output:
```
Testing camera configuration loading...
Camera serials: ['317422074281', '327122079374']
...
✅ Configuration loaded successfully!
```

## Dependencies

- **PyYAML**: Already installed in your `.venv`
- No additional dependencies needed

## Tips

1. **Backup the config**: Before making major changes, copy `configs/camera_config.yaml`
2. **Comment your changes**: Add comments in YAML explaining why you changed values
3. **Test after changes**: Run a simple script to verify configuration loads correctly
4. **Keep it organized**: Group related settings together in the YAML file

## Example Workflow

```bash
# 1. Edit configuration
nano configs/camera_config.yaml

# 2. Change what you need (e.g., laser power, exposure, etc.)

# 3. Run any script - it automatically uses new config!
python3 cap/segment_pc.py
```

No code changes needed!
