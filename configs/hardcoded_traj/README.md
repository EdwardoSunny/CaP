# Hardcoded trajectory files

Each `<method>.txt` file in this directory is replayed verbatim whenever the
matching adapter method fires. The filename (without `.txt`) IS the method
name — it must match the `method:` field of an entry in
`../action_map.yaml`.

## Format

One step per line. Comments with `#`. Blank lines ignored.

```
# Comment
x, y, z, rx, ry, rz             # move to pose (mm + deg)
x, y, z, rx, ry, rz, duration   # ... with custom duration (seconds)
OPEN                            # open gripper
CLOSE                           # close gripper
WAIT 0.5                        # sleep 0.5s
END                             # stop parsing (everything after is ignored)
```

Units: position in **mm**, orientation (rpy) in **degrees**. Poses are
absolute, in the robot base frame.

## Example (`put_in.txt`)

```
# Drop held object into container
393.2, 0.0, 250.0, 180.0, 0.0, 0.0, 3.0
393.2, 0.0, 100.0, 180.0, 0.0, 0.0, 2.0
OPEN
WAIT 0.5
393.2, 0.0, 250.0, 180.0, 0.0, 0.0, 2.0
END
```

## Runtime wiring

In `main.py`:

```python
setup_LMP(
    config, env, xarm_config,
    plan_mode="json",
    hardcoded_traj_dir="configs/hardcoded_traj",
)
```

On startup, `HardcodedActionAdapter` scans this directory and installs one
method override per `.txt` file. Methods with no matching file keep the
parent `VirtualHomeActionAdapter`'s real implementation.
