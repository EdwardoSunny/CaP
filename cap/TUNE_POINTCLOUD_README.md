# Point Cloud Alignment Tuning Tool

## Overview

The `tune_pointcloud.py` tool allows you to manually fine-tune the alignment between two point clouds after applying the base calibration (`transforms.npy`). This is useful when there are small residual offsets that need manual adjustment.

## How It Works

1. **Captures point clouds** from both cameras
2. **Applies base transforms** from `transforms/transforms.npy`
3. **Visualizes alignment** with Camera 1 (red) and Camera 2 (green)
4. **Allows interactive adjustment** of Camera 2's position and rotation
5. **Saves manual offset** to `transforms/manual_offset.npy`
6. **Automatically applied** by `segment_pc.py` and `pc.py` when the offset file exists

## Usage

### 1. Run the Tuning Tool

```bash
cd /home/u-ril/edward/CaP
python3 cap/tune_pointcloud.py
```

### 2. Interactive Commands

Once the tool starts, you'll see both point clouds:
- **Red**: Camera 1 (317422074281) - base/reference
- **Green**: Camera 2 (327122079374) - will be adjusted

Available commands:

- **`view`** - Show current alignment in 3D viewer
- **`set <param> <value>`** - Set a parameter to a specific value
  - Example: `set tx 0.01` (set x translation to 0.01 meters)
- **`adjust <param> <delta>`** - Adjust a parameter by a delta
  - Example: `adjust tx 0.001` (add 0.001 meters to x translation)
- **`reset`** - Reset all adjustments to zero
- **`save`** - Save current offset and exit
- **`quit`** - Exit without saving

### 3. Adjustment Parameters

**Translation (in meters):**
- `tx` - X-axis translation
- `ty` - Y-axis translation
- `tz` - Z-axis translation

**Rotation (in degrees):**
- `rx` - Rotation around X-axis
- `ry` - Rotation around Y-axis
- `rz` - Rotation around Z-axis

### 4. Typical Workflow

```
> view                    # See initial alignment
> adjust tx 0.005         # Move green cloud 5mm in +X
> adjust tz -0.003        # Move green cloud 3mm in -Z
> view                    # Check the adjustment
> adjust rz 0.5           # Rotate 0.5 degrees around Z
> view                    # Check again
> save                    # Save when satisfied
```

### 5. Using the Saved Offset

Once you save the offset to `transforms/manual_offset.npy`, it will be **automatically applied** when you run:

- `cap/segment_pc.py` - Segmentation with point cloud merging
- `cap/pc.py` - Basic point cloud merging

The offset is applied **after** the base calibration transform and any ICP refinement.

## File Locations

- **Input**: `transforms/transforms.npy` - Base calibration (required)
- **Output**: `transforms/manual_offset.npy` - Manual fine-tuning offset (created by this tool)

## Tips

1. **Start small**: Use small increments (0.001-0.005 for translation, 0.1-1.0 for rotation)
2. **View often**: Use `view` frequently to check your adjustments
3. **Focus on one axis**: Adjust one parameter at a time for clarity
4. **Translation units**: Remember that translation is in meters (0.001 = 1mm)
5. **Rotation order**: Rotations are applied in ZYX order (Euler angles)

## Example Session

```bash
$ python3 cap/tune_pointcloud.py

Configuring cameras...
Loaded transforms for cameras: ['317422074281', '327122079374']
Capturing point clouds...

Camera 1 (317422074281): 45231 points
Camera 2 (327122079374): 43892 points

Interactive tuning interface ready!

> view
[3D window opens showing red and green point clouds]

> adjust tx 0.003
Adjusted tx by 0.003, new value: 0.003

> adjust ty -0.002
Adjusted ty by -0.002, new value: -0.002

> view
[Check alignment - looking better!]

> adjust rz 0.8
Adjusted rz by 0.8, new value: 0.8

> view
[Perfect alignment achieved!]

> save
Manual offset saved to transforms/manual_offset.npy
Parameters: {'tx': 0.003, 'ty': -0.002, 'tz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.8}

This offset will be automatically applied when you run segment_pc.py or pc.py
```

## Removing the Manual Offset

If you want to disable the manual offset:

```bash
rm transforms/manual_offset.npy
```

The scripts will automatically detect its absence and skip applying it.
