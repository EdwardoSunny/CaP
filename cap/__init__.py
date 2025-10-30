"""
Cap - Multi-Camera Point Cloud Capture Package

This package provides tools for capturing and merging point clouds from
multiple RealSense cameras, with optional AI-powered segmentation.
"""

from .pc import RobotFrameMerger as BasicMerger

__all__ = ["BasicMerger"]
__version__ = "1.0.0"
