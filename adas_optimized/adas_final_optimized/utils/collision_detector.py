"""
Collision Detection System
Warns about potential collisions
"""

import numpy as np
from typing import List, Dict, Tuple
from enum import Enum

class WarningLevel(Enum):
    """Collision warning levels"""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class CollisionDetector:
    """Detect potential collisions with tracked objects"""
    
    def __init__(self):
        # Distance thresholds (meters)
        self.critical_distance = 5.0
        self.warning_distance = 10.0
        self.caution_distance = 20.0
        
        # Lane width assumption (meters)
        self.lane_width = 3.5
        
    def check_collision_risk(self,
                            tracks: List,
                            distances: Dict[int, float],
                            lane_left: np.ndarray = None,
                            lane_right: np.ndarray = None,
                            img_width: int = 1280,
                            road_mask: np.ndarray = None) -> Dict:
        """
        Check collision risk with all tracked objects
        
        Args:
            tracks: List of tracked objects
            distances: Dict mapping track_id to distance
            lane_left: Left lane points
            lane_right: Right lane points
            img_width: Image width
            road_mask: Optional road segmentation mask (ignores off-road objects)
        
        Returns:
            Dict with collision analysis
        """
        result = {
            'overall_level': WarningLevel.SAFE,
            'critical_objects': [],
            'warning_objects': [],
            'caution_objects': [],
            'closest_distance': float('inf'),
            'in_path_objects': []
        }
        
        if not tracks:
            return result
        
        # Calculate vehicle path (center of lane)
        vehicle_center_x = img_width / 2
        
        for track in tracks:
            track_id = track.id
            distance = distances.get(track_id, -1)
            
            if distance < 0:
                continue
            
            # Check if object is in our path
            x1, y1, x2, y2 = track.bbox
            obj_center_x = (x1 + x2) / 2
            obj_center_y = (y1 + y2) / 2
            
            # If road mask available, check if object is on road
            on_road = True
            if road_mask is not None:
                try:
                    # Check if center point is on road
                    if 0 <= int(obj_center_y) < road_mask.shape[0] and \
                       0 <= int(obj_center_x) < road_mask.shape[1]:
                        on_road = road_mask[int(obj_center_y), int(obj_center_x)] > 0
                    else:
                        on_road = False
                except:
                    on_road = True  # Default to True if check fails
            
            # Skip objects not on road (e.g., parked cars on sidewalk)
            if not on_road:
                continue
            
            # Check lateral offset from vehicle center
            lateral_offset = abs(obj_center_x - vehicle_center_x)
            lateral_offset_m = (lateral_offset / img_width) * 3.5  # Assume 3.5m lane width
            
            # Object is in path if within lane width
            in_path = lateral_offset_m < (self.lane_width / 2)
            
            obj_info = {
                'track_id': track_id,
                'distance': distance,
                'lateral_offset': lateral_offset_m,
                'in_path': in_path,
                'bbox': track.bbox,
                'class_id': track.class_id,
                'class_name': getattr(track, 'class_name', 'object')  # Add class name
            }
            
            # Categorize by distance
            if distance < self.critical_distance and in_path:
                result['critical_objects'].append(obj_info)
                result['overall_level'] = WarningLevel.CRITICAL
                
            elif distance < self.warning_distance and in_path:
                result['warning_objects'].append(obj_info)
                if result['overall_level'] != WarningLevel.CRITICAL:
                    result['overall_level'] = WarningLevel.WARNING
                    
            elif distance < self.caution_distance and in_path:
                result['caution_objects'].append(obj_info)
                if result['overall_level'] == WarningLevel.SAFE:
                    result['overall_level'] = WarningLevel.CAUTION
            
            if in_path:
                result['in_path_objects'].append(obj_info)
            
            # Track closest object
            if distance < result['closest_distance']:
                result['closest_distance'] = distance
        
        # Sort by distance
        result['in_path_objects'].sort(key=lambda x: x['distance'])
        
        return result
    
    def get_warning_message(self, collision_result: Dict) -> str:
        """Generate warning message (ASCII-safe)"""
        level = collision_result['overall_level']
        
        if level == WarningLevel.CRITICAL:
            return "[!] CRITICAL: COLLISION IMMINENT!"
        elif level == WarningLevel.WARNING:
            return "[!] WARNING: Object ahead"
        elif level == WarningLevel.CAUTION:
            return "[!] CAUTION: Monitor ahead"
        else:
            return "[OK] Safe - No immediate threats"
    
    def get_warning_color(self, level: WarningLevel) -> Tuple[int, int, int]:
        """Get BGR color for warning level"""
        if level == WarningLevel.CRITICAL:
            return (0, 0, 255)  # Red
        elif level == WarningLevel.WARNING:
            return (0, 165, 255)  # Orange
        elif level == WarningLevel.CAUTION:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green
