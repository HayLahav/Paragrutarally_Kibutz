"""
Lane Detector with SegFormer Integration
Combines traditional CV with semantic segmentation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

class LaneDetectorWithSegmentation:
    """Lane detector that can use SegFormer or traditional CV"""
    
    def __init__(self, segformer=None):
        """
        Initialize lane detector
        
        Args:
            segformer: Optional SegFormer segmentation model
        """
        self.segformer = segformer
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.prev_road_mask = None
        self.prev_safe_path = None
        
        # Mode tracking
        self.lane_mode = True
        self.no_lane_counter = 0
        
        # Segmentation cache
        self.cached_seg_result = None
        
    def detect(self, frame: np.ndarray, seg_result: Optional[Dict] = None) -> Tuple:
        """
        Detect lanes using best available method
        
        Args:
            frame: Input BGR frame
            seg_result: Optional pre-computed SegFormer segmentation
            
        Returns:
            (left_lane, right_lane, confidence, mode, extra_data)
        """
        h, w = frame.shape[:2]
        
        # Try SegFormer first if available
        if seg_result is not None or (self.segformer is not None and seg_result is None):
            return self._detect_with_segformer(frame, seg_result)
        
        # Fall back to traditional CV
        return self._detect_traditional(frame)
    
    def _detect_with_segformer(self, frame: np.ndarray, seg_result: Optional[Dict]) -> Tuple:
        """Detect using SegFormer segmentation"""
        h, w = frame.shape[:2]
        
        # Use provided result or get from BiSeNet
        if seg_result is None and self.segformer:
            seg_result = self.cached_seg_result or self.segformer.segment(frame)
            self.cached_seg_result = seg_result
        
        if seg_result is None:
            return self._detect_traditional(frame)
        
        road_mask = seg_result.get('road_mask')
        
        if road_mask is None:
            return self._detect_traditional(frame)
        
        # Extract boundaries from segmentation
        left_boundary, right_boundary = self._boundaries_from_mask(road_mask, h)
        
        # Get safe navigation path
        obstacle_map = seg_result.get('obstacle_map')
        safe_path = self._get_safe_path_from_mask(road_mask, obstacle_map, h)
        
        # Package extra data
        extra_data = {
            'road_mask': road_mask,
            'parking_mask': seg_result.get('parking_mask'),
            'obstacle_map': obstacle_map,
            'safe_path': safe_path,
            'segmentation_map': seg_result.get('segmentation_map'),
            'colored_map': seg_result.get('colored_map'),
            'road_confidence': seg_result.get('confidence', 0.8)
        }
        
        confidence = seg_result.get('confidence', 0.8)
        
        if left_boundary is not None and right_boundary is not None:
            self.prev_left_lane = left_boundary
            self.prev_right_lane = right_boundary
            self.prev_safe_path = safe_path
            return left_boundary, right_boundary, confidence, 'segformer', extra_data
        
        return self._detect_traditional(frame)
    
    def _detect_traditional(self, frame: np.ndarray) -> Tuple:
        """Traditional CV-based detection"""
        h, w = frame.shape[:2]
        
        # Try lane detection first
        left_lane, right_lane, lane_confidence = self._detect_lanes(frame)
        
        extra_data = {
            'road_mask': None,
            'safe_path': None,
            'road_confidence': lane_confidence
        }
        
        if lane_confidence > 0.6:
            self.no_lane_counter = 0
            self.lane_mode = True
            
            if left_lane is not None:
                self.prev_left_lane = left_lane
            if right_lane is not None:
                self.prev_right_lane = right_lane
            
            return left_lane, right_lane, lane_confidence, 'lanes', extra_data
        
        # Try drivable area
        self.no_lane_counter += 1
        
        if self.no_lane_counter > 5:
            self.lane_mode = False
            left_boundary, right_boundary, area_confidence = self._detect_drivable_area(frame)
            
            if area_confidence > 0.4:
                self.prev_left_lane = left_boundary
                self.prev_right_lane = right_boundary
                return left_boundary, right_boundary, area_confidence, 'drivable_area', extra_data
        
        return self.prev_left_lane, self.prev_right_lane, 0.3, 'fallback', extra_data
    
    def _boundaries_from_mask(self, road_mask: np.ndarray, h: int) -> Tuple:
        """Extract lane boundaries from road segmentation mask"""
        if road_mask is None:
            return None, None
        
        h_mask, w_mask = road_mask.shape
        left_points = []
        right_points = []
        
        y_start = int(h * 0.5)
        for y in range(y_start, h, 5):
            if y >= h_mask:
                continue
            
            row = road_mask[y, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 10:
                left_edge = road_pixels[0]
                right_edge = road_pixels[-1]
                
                left_points.append([left_edge, y])
                right_points.append([right_edge, y])
        
        if len(left_points) < 5 or len(right_points) < 5:
            return None, None
        
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        try:
            z_left = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            y_vals = np.linspace(y_start, h, 30)
            x_left = z_left[0] * y_vals**2 + z_left[1] * y_vals + z_left[2]
            left_boundary = np.column_stack([x_left, y_vals])
            
            z_right = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
            x_right = z_right[0] * y_vals**2 + z_right[1] * y_vals + z_right[2]
            right_boundary = np.column_stack([x_right, y_vals])
            
            return left_boundary, right_boundary
        except:
            return None, None
    
    def _get_safe_path_from_mask(self, road_mask: np.ndarray, 
                                 obstacle_map: Optional[np.ndarray], h: int) -> Optional[np.ndarray]:
        """Calculate safe path from road mask, avoiding obstacles"""
        if road_mask is None:
            return None
        
        # Erode road mask around obstacles
        if obstacle_map is not None:
            kernel = np.ones((15, 15), np.uint8)
            obstacles_dilated = cv2.dilate(obstacle_map, kernel, iterations=2)
            safe_road = cv2.bitwise_and(road_mask, cv2.bitwise_not(obstacles_dilated))
        else:
            safe_road = road_mask
        
        path_points = []
        y_start = int(h * 0.5)
        
        for y in range(y_start, h, 5):
            if y >= safe_road.shape[0]:
                continue
            
            row = safe_road[y, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 5:
                center_x = int(np.median(road_pixels))
                path_points.append([center_x, y])
        
        if len(path_points) < 5:
            return None
        
        path_points = np.array(path_points)
        
        try:
            z = np.polyfit(path_points[:, 1], path_points[:, 0], 2)
            y_vals = np.linspace(y_start, h, 30)
            x_vals = z[0] * y_vals**2 + z[1] * y_vals + z[2]
            safe_path = np.column_stack([x_vals, y_vals])
            return safe_path
        except:
            return None
    
    def _detect_lanes(self, frame: np.ndarray) -> Tuple:
        """Traditional lane line detection"""
        h, w = frame.shape[:2]
        
        roi_vertices = np.array([[
            (int(w * 0.1), h),
            (int(w * 0.4), int(h * 0.6)),
            (int(w * 0.6), int(h * 0.6)),
            (int(w * 0.9), h)
        ]], dtype=np.int32)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180,
                               threshold=50, minLineLength=50, maxLineGap=100)
        
        if lines is None:
            return None, None, 0.0
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])
        
        left_lane = self._fit_lane_line(left_lines, h)
        right_lane = self._fit_lane_line(right_lines, h)
        
        confidence = 0.8 if (left_lane is not None and right_lane is not None) else 0.4
        
        return left_lane, right_lane, confidence
    
    def _detect_drivable_area(self, frame: np.ndarray) -> Tuple:
        """Fallback drivable area detection"""
        h, w = frame.shape[:2]
        
        roi_top = int(h * 0.4)
        roi = frame[roi_top:h, :]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = np.abs(sobelx)
        sobelx = np.uint8(sobelx / sobelx.max() * 255)
        
        _, thresh = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)
        
        left_points = []
        right_points = []
        
        for y in range(0, roi.shape[0], 10):
            left_row = thresh[y, :w//2]
            right_row = thresh[y, w//2:]
            
            left_edges = np.where(left_row > 0)[0]
            right_edges = np.where(right_row > 0)[0]
            
            if len(left_edges) > 0:
                left_points.append([left_edges[0], y + roi_top])
            
            if len(right_edges) > 0:
                right_points.append([right_edges[-1] + w//2, y + roi_top])
        
        if len(left_points) < 5 or len(right_points) < 5:
            return None, None, 0.0
        
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        try:
            z_left = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            z_right = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
            
            y_vals = np.linspace(roi_top, h, 30)
            x_left = z_left[0] * y_vals**2 + z_left[1] * y_vals + z_left[2]
            x_right = z_right[0] * y_vals**2 + z_right[1] * y_vals + z_right[2]
            
            left_boundary = np.column_stack([x_left, y_vals])
            right_boundary = np.column_stack([x_right, y_vals])
            
            return left_boundary, right_boundary, 0.7
        except:
            return None, None, 0.0
    
    def _fit_lane_line(self, lines, img_height):
        """Fit polynomial to lane points"""
        if not lines:
            return None
        
        points = []
        for x1, y1, x2, y2 in lines:
            points.extend([(x1, y1), (x2, y2)])
        
        if len(points) < 2:
            return None
        
        points = np.array(points)
        
        try:
            z = np.polyfit(points[:, 1], points[:, 0], 2)
            y_points = np.linspace(img_height * 0.6, img_height, 30)
            x_points = z[0] * y_points**2 + z[1] * y_points + z[2]
            lane_points = np.column_stack([x_points, y_points])
            return lane_points
        except:
            return None
