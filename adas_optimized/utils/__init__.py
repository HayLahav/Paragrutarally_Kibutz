"""
Optimized ADAS System Utilities - Custom 1080p Version
Performance-optimized with Moondream2 VLM preparation
"""

from .byte_tracker import ByteTracker, Track
from .lane_detector import LaneDetectorWithSegmentation
from .distance_estimator import DistanceEstimator
from .collision_detector import CollisionDetector, WarningLevel
from .video_stabilizer import FastVideoStabilizer, SimpleVideoStabilizer, VideoStabilizer
from .segformer_segmentation import SegFormerSegmentation

# Removed for Moondream2 VLM:
# - BumperDetector (will use VLM hazard detection)
# - LicensePlateDetector (will use VLM vehicle details)
# - SpeedBumpDetector (will use VLM road conditions)

__all__ = [
    'ByteTracker',
    'Track',
    'LaneDetectorWithSegmentation',
    'DistanceEstimator',
    'CollisionDetector',
    'WarningLevel',
    'FastVideoStabilizer',
    'SimpleVideoStabilizer',
    'VideoStabilizer',
    'SegFormerSegmentation'
]
