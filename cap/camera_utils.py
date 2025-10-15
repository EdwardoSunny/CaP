#!/usr/bin/env python3
"""
Utility functions for camera configuration and setup.
Centralized configuration loading and camera initialization.
"""

import yaml
import os
import pyrealsense2 as rs


def load_camera_config(config_path="configs/camera_config.yaml"):
    """Load camera configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Camera config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def configure_realsense_cameras(config=None):
    """
    Configure RealSense cameras based on configuration.

    Args:
        config: Configuration dict. If None, loads from camera_config.yaml
    """
    if config is None:
        config = load_camera_config()

    camera_serials = config['camera_serials']
    camera_settings = config['camera_settings']
    rgb_settings = config['rgb_settings']

    ctx = rs.context()
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)

        if serial not in camera_serials:
            continue

        settings = camera_settings.get(serial)
        if not settings:
            print(f"Warning: No settings found for camera {serial}, skipping")
            continue

        print(f"\nConfiguring camera {serial}...")

        # Enable Advanced Mode
        adv = rs.rs400_advanced_mode(dev)
        if not adv.is_enabled():
            adv.toggle_advanced_mode(True)
            print(f"  Advanced mode enabled")

        # Find sensors
        stereo = next(
            s for s in dev.query_sensors()
            if "Stereo" in s.get_info(rs.camera_info.name)
        )
        rgb_sensor = next(
            (s for s in dev.query_sensors()
             if "RGB" in s.get_info(rs.camera_info.name)),
            None
        )

        # === Depth/Stereo Module Configuration ===
        stereo.set_option(rs.option.enable_auto_exposure, settings['auto_exposure'])
        print(f"  Depth auto exposure: {'ON' if settings['auto_exposure'] else 'OFF'}")

        if settings['auto_exposure'] == 0 and settings.get('exposure') is not None:
            stereo.set_option(rs.option.exposure, settings['exposure'])
            print(f"  Depth exposure: {settings['exposure']} µs")

        stereo.set_option(rs.option.gain, settings['gain'])
        print(f"  Depth gain: {settings['gain']}")

        stereo.set_option(rs.option.laser_power, settings['laser_power'])
        print(f"  Laser power: {settings['laser_power']} mW")

        # === RGB Sensor Configuration ===
        if rgb_sensor:
            rgb_sensor.set_option(
                rs.option.enable_auto_white_balance,
                rgb_settings['enable_auto_white_balance']
            )
            print(f"  RGB auto white balance: {'ON' if rgb_settings['enable_auto_white_balance'] else 'OFF'}")

            rgb_sensor.set_option(
                rs.option.enable_auto_exposure,
                rgb_settings['enable_auto_exposure']
            )
            print(f"  RGB auto exposure: {'ON' if rgb_settings['enable_auto_exposure'] else 'OFF'}")

        print(f"  Camera {serial} configured successfully")


def get_stream_config(config=None):
    """Get stream configuration from config file"""
    if config is None:
        config = load_camera_config()

    return config['stream_config']


def get_point_cloud_config(config=None):
    """Get point cloud processing configuration"""
    if config is None:
        config = load_camera_config()

    return config['point_cloud']


def get_calibration_config(config=None):
    """Get calibration file paths and settings"""
    if config is None:
        config = load_camera_config()

    return config['calibration']


def get_workspace_bounds(config=None):
    """Get workspace bounds for cropping"""
    if config is None:
        config = load_camera_config()

    ws = config['workspace_bounds']
    if ws['enabled']:
        return [ws['min_x'], ws['min_y'], ws['min_z'],
                ws['max_x'], ws['max_y'], ws['max_z']]
    return None


def get_segmentation_config(config=None):
    """Get segmentation settings"""
    if config is None:
        config = load_camera_config()

    return config['segmentation']


if __name__ == "__main__":
    # Test loading configuration
    print("Testing camera configuration loading...")
    config = load_camera_config()

    print(f"\nCamera serials: {config['camera_serials']}")
    print(f"\nCamera settings:")
    for serial, settings in config['camera_settings'].items():
        print(f"  {serial}: {settings}")

    print(f"\nStream config: {get_stream_config(config)}")
    print(f"\nPoint cloud config: {get_point_cloud_config(config)}")
    print(f"\nCalibration config: {get_calibration_config(config)}")
    print(f"\nWorkspace bounds: {get_workspace_bounds(config)}")
    print(f"\nSegmentation config: {get_segmentation_config(config)}")

    print("\n✅ Configuration loaded successfully!")
