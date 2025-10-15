# Point Cloud Alignment Tools Summary

You now have **3 tools** for point cloud alignment in the `cap/` directory:

## 1. 🎮 `tune_pointcloud_interactive.py` ⭐ **RECOMMENDED**

**Best for:** Quick, intuitive alignment with real-time visual feedback

### Features:
- **Real-time visual updates** - See changes instantly as you press keys
- **Arrow keys** for translation (↑↓←→ + PgUp/PgDn)
- **WASD+QE** for rotation
- **Original RGB colors** - Both clouds keep natural appearance
- **Live status display** - Shows current tx, ty, tz, rx, ry, rz values

### Usage:
```bash
python3 cap/tune_pointcloud_interactive.py
```

### Controls (in 3D viewer):
- **↑/↓** - Move X axis
- **←/→** - Move Y axis
- **PgUp/PgDn** - Move Z axis
- **W/S** - Rotate X (pitch)
- **A/D** - Rotate Y (yaw)
- **Q/E** - Rotate Z (roll)
- **+/-** - Adjust step size
- **R** - Reset
- **ESC** - Save & exit
- **X** - Exit without saving

---

## 2. 💬 `tune_pointcloud.py`

**Best for:** Text-based control if you prefer typing commands

### Features:
- Command-line interface
- Type commands like `set tx 0.01` or `adjust rz 0.5`
- Manual view window opening (close to continue)

### Usage:
```bash
python3 cap/tune_pointcloud.py
```

### Commands:
- `view` - Show alignment
- `set <param> <value>` - Set parameter
- `adjust <param> <delta>` - Adjust by delta
- `reset` - Reset all
- `save` - Save & exit
- `quit` - Exit without saving

---

## 3. 🔧 Manual editing in `segment_pc.py` / `pc.py`

**Best for:** When you know exact offset values

Just edit the code directly or use numpy to create the offset file.

---

## 📁 File Structure

### Input:
- `transforms/transforms.npy` - Base calibration (from camera calibration)

### Output:
- `transforms/manual_offset.npy` - Fine-tuning offset (created by tools above)

### Auto-applied in:
- `cap/segment_pc.py` - Segmentation pipeline
- `cap/pc.py` - Basic point cloud merging

---

## 🔄 Workflow

```
Step 1: Run calibration (already done)
  └─> Creates transforms/transforms.npy

Step 2: Fine-tune alignment
  └─> python3 cap/tune_pointcloud_interactive.py
  └─> Creates transforms/manual_offset.npy

Step 3: Use in your pipelines
  └─> python3 cap/segment_pc.py
  └─> Automatically applies both transforms!
```

---

## 🎯 Which Tool to Use?

| Scenario | Recommended Tool |
|----------|------------------|
| **First time tuning** | `tune_pointcloud_interactive.py` |
| **Quick adjustments** | `tune_pointcloud_interactive.py` |
| **Prefer typing** | `tune_pointcloud.py` |
| **Know exact values** | Edit offset file directly |

---

## 💡 Tips

1. **Start with small steps**: Default is 1mm translation, 0.5° rotation
2. **Use +/- keys**: Adjust step size for coarse/fine tuning
3. **Press keys repeatedly**: See cumulative effect in real-time
4. **Camera 2 moves**: Camera 1 stays fixed as reference
5. **RGB colors**: Both clouds show original colors from cameras

---

## 🗑️ Remove Manual Offset

To disable the manual offset:
```bash
rm transforms/manual_offset.npy
```

Scripts will automatically detect its absence and skip it.

---

## 📊 Offset is Applied To

The manual offset is applied **only to Camera 2** (serial: 327122079374) to align it with Camera 1 (serial: 317422074281).

Application order:
1. Camera frame → Robot frame (transforms.npy)
2. ICP refinement (if icp_tf.npy exists)
3. Manual offset (if manual_offset.npy exists) ← **This is what you're tuning**
