#!/usr/bin/env python3
"""
Complete ADAS System for Jetson Orin Nano
Main Entry Point - Full Integration

Features:
- YOLOv8 object detection
- ByteTrack multi-object tracking
- BiSeNet-V2 semantic segmentation
- Video stabilization
- Bumper detection
- Speed bump detection
- Lane/road detection
- Distance estimation
- Collision warnings
- Enhanced visualizations

Usage:
    python main.py <video_file>
    python main.py <video_file> --output result.mp4
    python main.py <video_file> --bisenet --stabilize
"""

import cv2
import numpy as np
import time
import sys
import os
import json
import argparse
import math

# Import utilities
from utils.byte_tracker import ByteTracker
from utils.lane_detector import LaneDetectorWithSegmentation
from utils.distance_estimator import DistanceEstimator
from utils.collision_detector import CollisionDetector, WarningLevel
from utils.video_stabilizer import SimpleVideoStabilizer
# from utils.bumper_detector import BumperDetector, LicensePlateDetector  # Removed - Moondream2 will handle
# from utils.speed_bump_detector import SpeedBumpDetector  # Disabled - will use Moondream2
from utils.segformer_segmentation import SegFormerSegmentation

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)


class ADASSystem:
    """Complete ADAS System with all features"""
    
    def __init__(self, focal_length=700.0, use_bisenet=False, use_stabilization=True, use_vlm=False, vlm_frame_skip=120):
        """Initialize complete ADAS system"""
        print("="*70)
        print("COMPLETE ADAS SYSTEM - INITIALIZATION")
        print("="*70)
        
        # Load YOLO
        print("\n[1/10] Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        self.yolo.to('cuda')
        print("       ✓ YOLOv8 loaded on CUDA")
        
        # Initialize ByteTrack
        print("[2/10] Initializing ByteTrack...")
        self.tracker = ByteTracker(
            track_thresh=0.6,   # Higher confidence required (reduce false positives)
            match_thresh=0.9,   # Tighter matching (less bbox spread)
            max_age=15          # Drop lost tracks faster
        )
        print("       ✓ Tracker initialized")
        
        # Initialize SegFormer (optional)
        self.use_segformer = use_bisenet
        if use_bisenet:
            print("[3/10] Initializing SegFormer...")
            try:
                self.segformer = SegFormerSegmentation(device='cuda')
                print("       ✓ SegFormer initialized")
            except Exception as e:
                print(f"       ⚠ SegFormer failed: {e}, using fallback")
                self.segformer = None
                self.use_segformer = False
        else:
            print("[3/10] SegFormer disabled (using traditional CV)")
            self.segformer = None
        
        # Initialize Lane Detector
        print("[4/10] Initializing Lane Detector...")
        self.lane_detector = LaneDetectorWithSegmentation(segformer=self.segformer)
        print("       ✓ Lane detector initialized")
        
        # Initialize Distance Estimator
        print("[5/10] Initializing Distance Estimator...")
        self.distance_estimator = DistanceEstimator(focal_length=focal_length)
        print(f"       ✓ Distance estimator initialized (f={focal_length})")
        
        # Initialize Collision Detector
        print("[6/10] Initializing Collision Detector...")
        self.collision_detector = CollisionDetector()
        print("       ✓ Collision detector initialized")
        
        # Initialize Video Stabilizer
        self.use_stabilization = use_stabilization
        if use_stabilization:
            print("[7/10] Initializing Video Stabilizer...")
            self.stabilizer = SimpleVideoStabilizer(smoothing_window=10)
            print("       ✓ Video stabilizer initialized")
        else:
            print("[7/10] Video stabilization disabled")
            self.stabilizer = None
        
        # Bumper Detector - DISABLED
        # Will be replaced with Moondream2 VLM
        print("[8/10] Bumper Detection disabled")
        print("        (Will use Moondream2 VLM in future)")
        self.bumper_detector = None
        
        # License Plate Detector - DISABLED
        print("[9/10] License Plate Detection disabled")
        print("        (Will use Moondream2 VLM in future)")
        self.plate_detector = None
        
        # Moondream2 VLM - OPTIONAL
        # Provides intelligent scene understanding and hazard detection
        self.use_vlm = use_vlm
        if use_vlm:
            print("[10/11] Initializing Moondream2 VLM...")
            try:
                from utils.moondream2_vlm import Moondream2Lite
                self.vlm = Moondream2Lite(frame_skip=vlm_frame_skip, max_tokens=32, verbose=False)
                if self.vlm.is_available():
                    print(f"        ✓ Moondream2 VLM initialized (frame_skip={vlm_frame_skip})")
                else:
                    print("        ⚠ Moondream2 not available, continuing without VLM")
                    self.vlm = None
                    self.use_vlm = False
            except Exception as e:
                print(f"        ⚠ VLM initialization failed: {e}")
                self.vlm = None
                self.use_vlm = False
        else:
            print("[10/11] Moondream2 VLM disabled")
            self.vlm = None
        
        # Speed Bump Detector - DISABLED
        # Now handled by Moondream2 VLM
        print("[11/11] Speed Bump Detection disabled")
        print("        (Replaced by Moondream2 VLM)")
        self.speed_bump_detector = None
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0.0
        self.component_times = {
            'stabilization': [],
            'yolo': [],
            'tracking': [],
            'segmentation': [],
            'lanes': [],
            'distance': [],
            'collision': [],
            'bumpers': [],
            'speed_bumps': [],
            'vlm': [],
            'steering': [],
            'visualization': []
        }
        
        # Visual enhancements
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("✓ SYSTEM READY")
        print("="*70 + "\n")
    
    def get_distance_color(self, distance):
        """Smooth gradient color based on distance"""
        if distance < 0:
            return (128, 128, 128)
        
        normalized = np.clip(distance / 20.0, 0, 1)
        
        if normalized < 0.5:
            r = 255
            g = int(255 * (normalized * 2))
            b = 0
        else:
            r = int(255 * (2 - normalized * 2))
            g = 255
            b = 0
        
        return (b, g, r)
    
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius=10):
        """Draw rectangle with rounded corners"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        radius = min(radius, abs(x2-x1)//2, abs(y2-y1)//2)
        
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_dashed_line(self, img, points, color, thickness, dash_length=20):
        """Draw a dashed line through points"""
        if len(points) < 2:
            return
        
        for i in range(len(points) - 1):
            total_dist = 0
            for j in range(i):
                total_dist += np.linalg.norm(points[j+1] - points[j])
            
            if int(total_dist / dash_length) % 2 == 0:
                cv2.line(img, tuple(points[i]), tuple(points[i+1]), color, thickness, cv2.LINE_AA)
    
    def draw_birds_eye_view(self, output, tracks, distances, w, h, extra_data):
        """Draw mini top-down view in corner"""
        map_w, map_h = 250, 180
        mini_map = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        
        # Draw road from segmentation if available
        road_mask = extra_data.get('road_mask')
        if road_mask is not None:
            road_small = cv2.resize(road_mask, (map_w, map_h))
            mini_map[road_small > 0] = [40, 80, 40]  # Dark green
        else:
            # Default road
            cv2.rectangle(mini_map, (60, 0), (190, map_h), (60, 60, 60), -1)
        
        # Draw lane lines
        cv2.line(mini_map, (85, 0), (85, map_h), (255, 255, 255), 1)
        cv2.line(mini_map, (125, 0), (125, map_h), (255, 255, 255), 2)
        cv2.line(mini_map, (165, 0), (165, map_h), (255, 255, 255), 1)
        
        # Draw vehicle
        vehicle_pts = np.array([
            [125, map_h-10], [115, map_h-5], [115, map_h],
            [135, map_h], [135, map_h-5]
        ], dtype=np.int32)
        cv2.fillPoly(mini_map, [vehicle_pts], (0, 255, 255))
        cv2.polylines(mini_map, [vehicle_pts], True, (255, 255, 255), 1)
        
        # Draw detected objects
        for track in tracks:
            dist = distances.get(track.id, -1)
            if dist > 0 and dist < 50:
                y_pos = int(map_h - 10 - (dist * 3))
                
                if y_pos < 0 or y_pos > map_h:
                    continue
                
                x1, _, x2, _ = track.bbox
                lateral = ((x1 + x2) / 2 - w/2) / w
                x_pos = int(125 + lateral * 120)
                x_pos = np.clip(x_pos, 70, 180)
                
                color = self.get_distance_color(dist)
                
                class_name = getattr(track, 'class_name', 'unknown')
                size = 6 if class_name in ['car', 'truck', 'bus'] else 3
                
                cv2.circle(mini_map, (x_pos, y_pos), size, color, -1)
                cv2.circle(mini_map, (x_pos, y_pos), size+1, (255, 255, 255), 1)
                
                if dist < 15:
                    cv2.putText(mini_map, f"{dist:.0f}", (x_pos+8, y_pos+4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw speed bumps on map
        speed_bumps = extra_data.get('speed_bumps', [])
        for bump in speed_bumps:
            # Project to map
            bump_y = int(map_h - (bump.get('distance', 20) * 3))
            if 0 <= bump_y < map_h:
                cv2.line(mini_map, (70, bump_y), (180, bump_y), (0, 255, 255), 2)
        
        cv2.rectangle(mini_map, (0, 0), (map_w-1, map_h-1), (100, 100, 100), 2)
        cv2.rectangle(mini_map, (0, 0), (map_w, 20), (0, 0, 0), -1)
        cv2.putText(mini_map, "BIRD'S EYE VIEW", (40, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        margin = 15
        output[margin:margin+map_h, w-map_w-margin:w-margin] = mini_map
    
    def draw_hud(self, output, results, w, h):
        """Tesla-inspired HUD with all info"""
        collision = results.get('collision', {})
        tracks = results.get('tracks', [])
        steering = results.get('steering_angle', 0.0)
        lane_conf = results.get('lane_confidence', 0.0)
        detection_mode = results.get('detection_mode', 'unknown')
        speed_bumps = results.get('speed_bumps', [])
        
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
        output = cv2.addWeighted(output, 0.4, overlay, 0.6, 0)
        
        # Left: System status
        status_color = (0, 255, 0) if lane_conf > 0.5 else (100, 100, 100)
        
        if detection_mode == 'lanes':
            mode_text = "LANE ASSIST"
        elif detection_mode == 'bisenet':
            mode_text = "AI VISION"
        elif detection_mode == 'drivable_area':
            mode_text = "AREA DETECT"
        else:
            mode_text = "AUTOPILOT"
        
        cv2.putText(output, mode_text, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        mode_indicator = "Lanes" if detection_mode == 'lanes' else "Area"
        cv2.putText(output, f"{mode_indicator}: {lane_conf:.0%}", (20, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Center: Main warning/status
        warning_level = collision.get('overall_level', WarningLevel.SAFE)
        warning_color = self.collision_detector.get_warning_color(warning_level)
        
        # Check for speed bump warning
        if len(speed_bumps) > 0:
            closest_bump = min(speed_bumps, key=lambda x: x.get('distance', 999))
            bump_dist = closest_bump.get('distance', 0)
            if bump_dist < 15:
                warning_text = f"BUMP {bump_dist:.0f}m"
                font_size = 1.0
                warning_color = (0, 255, 255)  # Yellow
            else:
                warning_text = self._get_standard_warning(collision)
                font_size = 0.9
        else:
            warning_text = self._get_standard_warning(collision)
            font_size = 1.0 if warning_level == WarningLevel.CRITICAL else 0.9
        
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(output, warning_text, (text_x, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, warning_color, 2)
        
        # Right: Object count and steering
        cv2.putText(output, f"{len(tracks)} Objects", (w-180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        direction = "LEFT" if steering < -5 else "RIGHT" if steering > 5 else "STRAIGHT"
        steer_color = (0, 165, 255) if abs(steering) > 10 else (200, 200, 200)
        cv2.putText(output, f"Steer: {direction}", (w-180, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, steer_color, 1)
        
        # VLM Hazard Display (if available)
        if 'vlm' in results and results['vlm']:
            vlm_data = results['vlm']
            if vlm_data and 'hazards' in vlm_data:
                hazard_text = vlm_data['hazards']
                
                # Create semi-transparent overlay below HUD
                vlm_overlay = output.copy()
                cv2.rectangle(vlm_overlay, (0, 75), (w, 115), (40, 0, 40), -1)
                output = cv2.addWeighted(output, 0.6, vlm_overlay, 0.4, 0)
                
                # Display VLM hazard text
                cv2.putText(output, "🔮 VLM:", (20, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 1)
                
                # Wrap text if too long
                max_width = w - 120
                if len(hazard_text) > 80:
                    hazard_text = hazard_text[:77] + "..."
                
                cv2.putText(output, hazard_text, (100, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def _get_standard_warning(self, collision):
        """Get standard warning text - more informative for driver"""
        warning_level = collision.get('overall_level', WarningLevel.SAFE)
        in_path_objects = collision.get('in_path_objects', [])
        
        if warning_level == WarningLevel.CRITICAL:
            # Better: Tell driver WHAT is ahead, not just to brake
            # They might be able to change lanes instead
            if in_path_objects:
                obj = in_path_objects[0]
                class_name = obj.get('class_name', 'OBJECT')
                distance = obj.get('distance', 0)
                if class_name.lower() in ['car', 'truck', 'bus']:
                    return f"CAR AHEAD {distance:.0f}m"
                elif class_name.lower() in ['person', 'pedestrian']:
                    return f"PERSON AHEAD {distance:.0f}m"
                elif class_name.lower() in ['motorcycle', 'bicycle']:
                    return f"BIKE AHEAD {distance:.0f}m"
                else:
                    return f"OBJECT AHEAD {distance:.0f}m"
            return "OBJECT AHEAD"
        elif warning_level == WarningLevel.WARNING:
            return "CAUTION"
        else:
            closest = collision.get('closest_distance', float('inf'))
            if closest < float('inf'):
                return f"{closest:.1f}m"
            return "CLEAR"
    
    def process_frame(self, frame):
        """Process single frame with all features"""
        frame_start = time.time()
        h, w = frame.shape[:2]
        
        results = {'frame_number': self.frame_count}
        
        # 0. Video Stabilization
        if self.stabilizer:
            t0 = time.time()
            frame, stab_stats = self.stabilizer.stabilize(frame)
            self.component_times['stabilization'].append((time.time() - t0) * 1000)
            results['stabilization'] = stab_stats
        
        # 1. YOLO Detection
        t0 = time.time()
        yolo_results = self.yolo(frame, conf=0.45, verbose=False)[0]  # Higher threshold = fewer false positives
        self.component_times['yolo'].append((time.time() - t0) * 1000)
        
        # 2. Object Tracking
        t0 = time.time()
        detections = []
        vehicle_boxes = []
        
        for box in yolo_results.boxes:
            det = {
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'score': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': yolo_results.names[int(box.cls[0])]
            }
            detections.append(det)
            
            if det['class_name'] in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_boxes.append(det)
        
        tracks = self.tracker.update(detections)
        
        # Add track IDs to vehicles
        for track in tracks:
            for veh in vehicle_boxes:
                if np.allclose(track.bbox, veh['bbox'], atol=5):
                    veh['track_id'] = track.id
                    break
        
        self.component_times['tracking'].append((time.time() - t0) * 1000)
        results['tracks'] = tracks
        results['num_objects'] = len(tracks)
        
        # 3. SegFormer Segmentation (if enabled, with frame-skip caching)
        seg_result = None
        if self.use_segformer and self.segformer:
            t0 = time.time()
            seg_result = self.segformer.segment(frame)  # Handles caching internally
            self.component_times['segmentation'].append((time.time() - t0) * 1000)
        
        # 4. Lane/Road Detection
        t0 = time.time()
        left_lane, right_lane, lane_confidence, detection_mode, extra_data = \
            self.lane_detector.detect(frame, seg_result)
        self.component_times['lanes'].append((time.time() - t0) * 1000)
        
        results['left_lane'] = left_lane
        results['right_lane'] = right_lane
        results['lane_confidence'] = lane_confidence
        results['detection_mode'] = detection_mode
        results['extra_data'] = extra_data
        
        # Add SegFormer cache status to extra_data
        if seg_result and 'cached' in seg_result:
            extra_data['segformer_cached'] = seg_result['cached']
        
        # 5. Speed Bump Detection - DISABLED
        # Will be replaced with Moondream2 VLM in future
        # Moondream2 will provide: road description + hazard detection + instructions
        # Example: "Clear road ahead with slight right curve. Watch for pedestrian crossing."
        t0 = time.time()
        # speed_bumps = self.speed_bump_detector.detect(
        #     frame, 
        #     extra_data.get('road_mask'),
        #     seg_result
        # )
        speed_bumps = []  # Disabled - will use Moondream2 VLM
        self.component_times['speed_bumps'].append((time.time() - t0) * 1000)
        results['speed_bumps'] = speed_bumps
        extra_data['speed_bumps'] = speed_bumps
        
        # 6. Bumper Detection - DISABLED
        # Will be replaced with Moondream2 VLM
        t0 = time.time()
        bumpers = []  # Disabled
        reflectors = []  # Disabled
        plates = []  # Disabled
        self.component_times['bumpers'].append((time.time() - t0) * 1000)
        
        results['bumpers'] = bumpers
        results['reflectors'] = reflectors
        results['license_plates'] = plates
        
        # 7. Distance Estimation (with bumper precision)
        t0 = time.time()
        distances = {}
        for track in tracks:
            class_name = track.class_name if hasattr(track, 'class_name') else 'unknown'
            
            # Check for bumper
            bumper_distance = None
            for bumper in bumpers:
                if bumper.get('vehicle_id') == track.id:
                    anchor_x, anchor_y = bumper['distance_anchor']
                    _, _, _, y2 = track.bbox
                    height = y2 - anchor_y
                    
                    if height > 0 and class_name in self.distance_estimator.OBJECT_SIZES:
                        real_height, _ = self.distance_estimator.OBJECT_SIZES[class_name]
                        bumper_distance = (real_height * self.distance_estimator.focal_length) / (height * 100)
                    break
            
            if bumper_distance and bumper_distance > 0:
                distances[track.id] = bumper_distance
            else:
                distance = self.distance_estimator.estimate(track.bbox, class_name)
                distances[track.id] = distance
        
        self.component_times['distance'].append((time.time() - t0) * 1000)
        results['distances'] = distances
        
        # 8. Steering Calculation
        t0 = time.time()
        safe_path = extra_data.get('safe_path')
        if safe_path is not None and len(safe_path) > 5:
            steering_angle = self._calculate_steering_from_path(safe_path, w, h)
        else:
            steering_angle = self._calculate_steering(left_lane, right_lane, w, h)
        self.component_times['steering'].append((time.time() - t0) * 1000)
        results['steering_angle'] = steering_angle
        
        # 9. Collision Detection
        t0 = time.time()
        collision_result = self.collision_detector.check_collision_risk(
            tracks=tracks,
            distances=distances,
            lane_left=left_lane,
            lane_right=right_lane,
            img_width=w,
            road_mask=extra_data.get('road_mask')
        )
        self.component_times['collision'].append((time.time() - t0) * 1000)
        results['collision'] = collision_result
        
        # 10. Moondream2 VLM Analysis (if enabled)
        t0 = time.time()
        if self.use_vlm and self.vlm and self.vlm.is_available():
            vlm_result = self.vlm.analyze_frame(frame, self.frame_count, tracks=tracks)
            results['vlm'] = vlm_result
        else:
            results['vlm'] = None
        self.component_times['vlm'].append((time.time() - t0) * 1000)
        
        # Total time
        results['processing_time'] = time.time() - frame_start
        self.total_time += results['processing_time']
        self.frame_count += 1
        
        return results
    
    def _calculate_steering(self, left_lane, right_lane, img_width, img_height):
        """Calculate steering from lanes"""
        if left_lane is None or right_lane is None:
            return 0.0
        
        try:
            num_points = min(len(left_lane), len(right_lane))
            if num_points < 2:
                return 0.0
            
            center_x = (left_lane[:num_points, 0] + right_lane[:num_points, 0]) / 2
            center_y = left_lane[:num_points, 1]
            
            vehicle_x = img_width / 2
            vehicle_y = img_height
            
            lookahead_y = img_height * 0.7
            idx = np.argmin(np.abs(center_y - lookahead_y))
            target_x = center_x[idx]
            
            dx = target_x - vehicle_x
            dy = vehicle_y - lookahead_y
            
            angle_rad = math.atan2(dx, dy)
            angle_deg = math.degrees(angle_rad)
            
            return np.clip(angle_deg, -35, 35)
        except:
            return 0.0
    
    def _calculate_steering_from_path(self, safe_path, img_width, img_height):
        """Calculate steering from safe path"""
        if safe_path is None or len(safe_path) < 5:
            return 0.0
        
        try:
            lookahead_y = img_height * 0.7
            idx = np.argmin(np.abs(safe_path[:, 1] - lookahead_y))
            target_x = safe_path[idx, 0]
            
            vehicle_x = img_width / 2
            dx = target_x - vehicle_x
            dy = img_height - lookahead_y
            
            angle_rad = math.atan2(dx, dy)
            angle_deg = math.degrees(angle_rad)
            
            return np.clip(angle_deg, -35, 35)
        except:
            return 0.0
    
    def visualize(self, frame, results):
        """Create complete visualization"""
        t0 = time.time()
        output = frame.copy()
        h, w = output.shape[:2]
        
        tracks = results.get('tracks', [])
        distances = results.get('distances', {})
        left_lane = results.get('left_lane')
        right_lane = results.get('right_lane')
        steering = results.get('steering_angle', 0.0)
        collision = results.get('collision', {})
        warning_level = collision.get('overall_level', WarningLevel.SAFE)
        in_path_ids = [obj['track_id'] for obj in collision.get('in_path_objects', [])]
        detection_mode = results.get('detection_mode', 'unknown')
        extra_data = results.get('extra_data', {})
        bumpers = results.get('bumpers', [])
        reflectors = results.get('reflectors', [])
        plates = results.get('license_plates', [])
        speed_bumps = results.get('speed_bumps', [])
        
        # 1. Draw road segmentation overlay (ROAD ONLY, not sidewalks)
        road_mask = extra_data.get('road_mask')
        if road_mask is not None:
            # Create purple overlay for road only
            road_overlay = np.zeros_like(output)
            road_overlay[road_mask > 0] = [128, 64, 128]  # Purple for road
            output = cv2.addWeighted(output, 0.7, road_overlay, 0.3, 0)
        
        # 2. Draw filled lane area
        if left_lane is not None and right_lane is not None:
            num_points = min(len(left_lane), len(right_lane))
            
            lane_poly = np.vstack([
                left_lane[:num_points],
                right_lane[:num_points][::-1]
            ]).astype(np.int32)
            
            if detection_mode == 'lanes':
                fill_color = (0, 200, 0)
                boundary_color = (0, 255, 0)
                center_color = (255, 255, 0)
            elif detection_mode == 'bisenet':
                fill_color = (200, 100, 0)
                boundary_color = (255, 150, 0)
                center_color = (255, 200, 0)
            else:
                fill_color = (0, 150, 200)
                boundary_color = (0, 200, 255)
                center_color = (0, 255, 255)
            
            # Filled polygon disabled for performance (~100ms savings)
            # overlay = output.copy()
            # cv2.fillPoly(overlay, [lane_poly], fill_color)
            # output = cv2.addWeighted(output, 0.75, overlay, 0.25, 0)
            
            if detection_mode == 'lanes':
                cv2.polylines(output, [left_lane.astype(np.int32)], False, boundary_color, 3)
                cv2.polylines(output, [right_lane.astype(np.int32)], False, boundary_color, 3)
            else:
                self.draw_dashed_line(output, left_lane.astype(np.int32), boundary_color, 3)
                self.draw_dashed_line(output, right_lane.astype(np.int32), boundary_color, 3)
            
            # CENTER LINE REMOVED - Only show lane boundaries
            # center_x = (left_lane[:num_points, 0] + right_lane[:num_points, 0]) / 2
            # center_y = left_lane[:num_points, 1]
            # center_pts = np.column_stack([center_x, center_y]).astype(np.int32)
            # cv2.polylines(output, [center_pts], False, center_color, 2, cv2.LINE_AA)
        
        # 3. SAFE PATH LINE REMOVED - Only show arrow indicator
        # safe_path = extra_data.get('safe_path')
        # if safe_path is not None and len(safe_path) > 5:
        #     cv2.polylines(output, [safe_path.astype(np.int32)], False, (255, 0, 255), 3, cv2.LINE_AA)
        
        # 4. Draw speed bumps
        for bump in speed_bumps:
            x1, y1, x2, y2 = map(int, bump['bbox'])
            conf = bump['confidence']
            
            # Yellow warning box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            label = f"SPEED BUMP: {conf:.1f}"
            cv2.putText(output, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw warning icon
            cv2.line(output, (x1+10, y1+10), (x2-10, y1+10), (0, 255, 255), 4)
            cv2.line(output, (x1+10, y2-10), (x2-10, y2-10), (0, 255, 255), 4)
        
        # 5. Draw tracked objects
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            distance = distances.get(track.id, -1)
            class_name = getattr(track, 'class_name', 'object')
            
            color = self.get_distance_color(distance)
            
            if track.id in in_path_ids and warning_level == WarningLevel.CRITICAL:
                pulse = abs(math.sin(self.frame_count * 0.3)) * 0.5 + 0.5
                thickness = int(3 + pulse * 3)
            elif track.id in in_path_ids:
                thickness = 4
            else:
                thickness = 2
            
            self.draw_rounded_rectangle(output, (x1, y1), (x2, y2), color, thickness, radius=12)
            
            if distance > 0:
                label = f"{class_name.capitalize()}: {distance:.1f}m"
                if track.id in in_path_ids:
                    label += " [!]"
            else:
                label = f"{class_name.capitalize()}"
            
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            label_bg_pts = np.array([
                [x1, y1-label_h-10], [x1+label_w+10, y1-label_h-10],
                [x1+label_w+10, y1-2], [x1, y1-2]
            ], dtype=np.int32)
            
            overlay = output.copy()
            cv2.fillPoly(overlay, [label_bg_pts], color)
            output = cv2.addWeighted(output, 0.6, overlay, 0.4, 0)
            
            cv2.putText(output, label, (x1+5, y1-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 6. Draw bumpers
        for bumper in bumpers:
            x1, y1, x2, y2 = map(int, bumper['bbox'])
            bumper_type = bumper['type']
            color = (255, 0, 255) if bumper_type == 'rear' else (255, 255, 0)
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            anchor_x, anchor_y = bumper['distance_anchor']
            cv2.circle(output, (anchor_x, anchor_y), 5, (0, 255, 0), -1)
            cv2.circle(output, (anchor_x, anchor_y), 8, color, 2)
            
            label = f"{bumper_type.upper()}"
            cv2.putText(output, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 7. Draw reflectors
        for reflector in reflectors:
            cx, cy = reflector['center']
            cv2.circle(output, (cx, cy), 3, (0, 0, 255), -1)
            cv2.circle(output, (cx, cy), 5, (255, 255, 255), 1)
        
        # 8. Draw license plates
        for plate in plates:
            x1, y1, x2, y2 = map(int, plate['bbox'])
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # 9. Draw vehicle position and steering
        vehicle_pos = (w // 2, h - 30)
        cv2.circle(output, vehicle_pos, 15, (0, 255, 255), -1)
        cv2.circle(output, vehicle_pos, 15, (255, 255, 255), 2)
        
        steering_rad = math.radians(steering)
        arrow_length = 80
        dx = int(arrow_length * math.sin(steering_rad))
        dy = int(-arrow_length * math.cos(steering_rad))
        
        arrow_color = (0, 0, 255) if abs(steering) > 15 else (0, 255, 255)
        cv2.arrowedLine(output, vehicle_pos,
                       (vehicle_pos[0] + dx, vehicle_pos[1] + dy),
                       arrow_color, 4, tipLength=0.3, line_type=cv2.LINE_AA)
        
        # 10. Flashing border for critical warnings
        if warning_level == WarningLevel.CRITICAL:
            flash = int((time.time() - self.start_time) * 4) % 2
            if flash:
                cv2.rectangle(output, (5, 5), (w-5, h-5), (0, 0, 255), 15)
                
                corner_size = 50
                cv2.line(output, (0, 0), (corner_size, 0), (0, 0, 255), 20)
                cv2.line(output, (0, 0), (0, corner_size), (0, 0, 255), 20)
                cv2.line(output, (w, 0), (w-corner_size, 0), (0, 0, 255), 20)
                cv2.line(output, (w, 0), (w, corner_size), (0, 0, 255), 20)
                cv2.line(output, (0, h), (corner_size, h), (0, 0, 255), 20)
                cv2.line(output, (0, h), (0, h-corner_size), (0, 0, 255), 20)
                cv2.line(output, (w, h), (w-corner_size, h), (0, 0, 255), 20)
                cv2.line(output, (w, h), (w, h-corner_size), (0, 0, 255), 20)
        
        # 11. HUD
        output = self.draw_hud(output, results, w, h)
        
        # 12. Bird's eye view
        self.draw_birds_eye_view(output, tracks, distances, w, h, extra_data)
        
        # 13. Bottom stats
        stats_overlay = output.copy()
        cv2.rectangle(stats_overlay, (0, h-50), (400, h), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.7, stats_overlay, 0.3, 0)
        
        fps = 1.0 / results['processing_time'] if results['processing_time'] > 0 else 0
        fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(output, f"FPS: {fps:.1f}", (10, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        if len(self.component_times['yolo']) > 0:
            yolo_time = np.mean(self.component_times['yolo'][-10:])
            track_time = np.mean(self.component_times['tracking'][-10:])
            lane_time = np.mean(self.component_times['lanes'][-10:])
            
            cv2.putText(output, f"Y:{yolo_time:.0f} T:{track_time:.0f} L:{lane_time:.0f}ms",
                       (10, h-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Show if SegFormer is active
        if self.use_segformer:
            seg_text = "SegFormer: ON"
            # Check if cached
            extra = results.get('extra_data', {})
            if extra.get('segformer_cached', False):
                seg_text += " (cached)"
            cv2.putText(output, seg_text, (200, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show if VLM is active
        if self.use_vlm and self.vlm:
            vlm_text = "VLM: ON"
            if 'vlm' in results and results['vlm']:
                if results['vlm']['cached']:
                    vlm_text += " (cached)"
            cv2.putText(output, vlm_text, (320, h-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        self.component_times['visualization'].append((time.time() - t0) * 1000)
        
        return output
    
    def get_statistics(self):
        """Get performance statistics"""
        stats = {
            'total_frames': self.frame_count,
            'total_time': self.total_time,
            'average_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
        }
        
        for component, times in self.component_times.items():
            if times:
                stats[f'{component}_avg_ms'] = float(np.mean(times))
                stats[f'{component}_max_ms'] = float(np.max(times))
        
        return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Complete ADAS System')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--focal-length', type=float, default=700.0,
                       help='Camera focal length (default: 700)')
    parser.add_argument('--output', help='Save output video to file')
    parser.add_argument('--stats', help='Save statistics to JSON file')
    parser.add_argument('--bisenet', action='store_true',
                       help='Enable SegFormer segmentation (auto-downloads from Hugging Face)')
    parser.add_argument('--no-stabilization', action='store_true',
                       help='Disable video stabilization')
    parser.add_argument('--vlm', action='store_true',
                       help='Enable Moondream2 VLM for scene understanding (requires run_moondream_image.py)')
    parser.add_argument('--vlm-skip', type=int, default=120,
                       help='VLM frame skip interval (default: 120 = 4 sec at 30fps, faster performance)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Initialize system
    system = ADASSystem(
        focal_length=args.focal_length,
        use_bisenet=args.bisenet,
        use_stabilization=not args.no_stabilization,
        use_vlm=args.vlm,
        vlm_frame_skip=args.vlm_skip
    )
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {args.video}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
    print("\nProcessing... (Press 'q' to quit)\n")
    
    # Setup output
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}\n")
    
    # Process video
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = system.process_frame(frame)
            output = system.visualize(frame, results)
            
            if writer:
                writer.write(output)
            
            if not args.output:
                try:
                    cv2.imshow("ADAS System - Press 'q' to quit", output)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nQuitting...")
                        break
                except:
                    pass
            
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    # Print statistics
    stats = system.get_statistics()
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Average FPS: {stats['average_fps']:.2f}")
    print("\nComponent Breakdown:")
    if stats.get('stabilization_avg_ms'):
        print(f"  Stabilization: {stats.get('stabilization_avg_ms', 0):.2f} ms")
    print(f"  YOLO:          {stats.get('yolo_avg_ms', 0):.2f} ms")
    print(f"  Tracking:      {stats.get('tracking_avg_ms', 0):.2f} ms")
    if stats.get('segmentation_avg_ms'):
        print(f"  SegFormer:     {stats.get('segmentation_avg_ms', 0):.2f} ms")
    print(f"  Lane:          {stats.get('lanes_avg_ms', 0):.2f} ms")
    print(f"  Distance:      {stats.get('distance_avg_ms', 0):.2f} ms")
    print(f"  Bumpers:       {stats.get('bumpers_avg_ms', 0):.2f} ms")
    print(f"  Speed Bumps:   {stats.get('speed_bumps_avg_ms', 0):.2f} ms")
    print(f"  Collision:     {stats.get('collision_avg_ms', 0):.2f} ms")
    if stats.get('vlm_avg_ms'):
        print(f"  VLM:           {stats.get('vlm_avg_ms', 0):.2f} ms")
    print(f"  Steering:      {stats.get('steering_avg_ms', 0):.2f} ms")
    print(f"  Visualization: {stats.get('visualization_avg_ms', 0):.2f} ms")
    print("="*70)
    
    if args.stats:
        with open(args.stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.stats}")
    
    print("\n✓ Processing complete!\n")


if __name__ == "__main__":
    main()
