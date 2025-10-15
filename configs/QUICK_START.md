# Quick Start - Camera Configuration

## 📁 Location
All camera configuration is in: **`configs/camera_config.yaml`**

## ⚡ Quick Edits

### Change what object to segment:
```bash
nano configs/camera_config.yaml
# Find: text_prompt: "Gray coffee machine"
# Change to: text_prompt: "Your object here"
```

### Adjust camera exposure:
```bash
nano configs/camera_config.yaml
# Find camera "327122079374" under camera_settings
# Change: exposure: 500  (to your desired value)
```

### Change laser power:
```bash
nano configs/camera_config.yaml
# Find your camera under camera_settings
# Change: laser_power: 240  (0-360 range)
```

## 🚀 Scripts That Use This Config

All these automatically read from `configs/camera_config.yaml`:
- `cap/tune_pointcloud_interactive.py` - Interactive alignment
- `cap/tune_pointcloud.py` - Text-based alignment
- `cap/segment_pc.py` - Point cloud segmentation
- `cap/pc.py` - Basic point cloud merging

## 💡 Pro Tip

**No code changes needed!** Just edit the YAML file and run your script.

```bash
# Edit config
nano configs/camera_config.yaml

# Run any script - it automatically uses new settings
python3 cap/segment_pc.py
```

## 📖 Full Documentation

See `configs/CONFIG_SYSTEM_README.md` for complete details.
