"""
SegFormer Semantic Segmentation for ADAS
Using Hugging Face pretrained model on Cityscapes
SegFormer is faster and more accurate than BiSeNet-V2!
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Dict

try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ transformers not installed - segmentation will use fallback mode")


class SegFormerSegmentation:
    """
    SegFormer for real-time road segmentation
    Trained on Cityscapes dataset (urban driving scenes)
    Segments: road, sidewalk, vehicles, people, etc.
    """
    
    def __init__(self, model_name: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 
                 device: str = 'cuda'):
        """
        Initialize SegFormer segmentation
        
        Args:
            model_name: Hugging Face model name
                - segformer-b0: Fastest (178 FPS on GPU)  <- Default
                - segformer-b1: Balanced (97 FPS)
                - segformer-b2: Better (61 FPS)
                - segformer-b4: Best for Jetson (33 FPS)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠ Transformers not available, using fallback segmentation")
            self.model = None
            self.processor = None
            return
        
        print(f"Loading SegFormer model: {model_name}")
        
        try:
            # Load model and processor
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            
            self.model.to(device)
            self.model.eval()
            
            print(f"✓ SegFormer loaded on {device}")
            
        except Exception as e:
            print(f"⚠ Failed to load SegFormer: {e}")
            print("  Falling back to traditional CV")
            self.model = None
            self.processor = None
            return
        
        # Cityscapes class definitions (19 classes)
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
            'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        # Define useful class groups
        self.drivable_classes = [0]  # road
        self.parking_classes = [1]   # sidewalk (can be parking)
        self.obstacle_classes = [11, 12, 13, 14, 15, 16, 17, 18]  # vehicles, people
        
        # Frame skip caching (process every N frames)
        self.frame_skip = 3  # Process every 3 frames
        self.frame_count = 0
        self.cached_result = None
        
        # Color map for visualization (BGR format)
        self.color_map = self._create_color_map()
    
    def _create_color_map(self):
        """Create color map for visualization"""
        color_map = np.zeros((256, 3), dtype=np.uint8)
        
        # Cityscapes standard colors (BGR)
        colors = [
            [128, 64, 128],   # road - purple
            [244, 35, 232],   # sidewalk - pink
            [70, 70, 70],     # building - dark gray
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light - orange
            [220, 220, 0],    # traffic sign - yellow
            [107, 142, 35],   # vegetation - green
            [152, 251, 152],  # terrain - light green
            [70, 130, 180],   # sky - blue
            [220, 20, 60],    # person - red
            [255, 0, 0],      # rider - bright red
            [0, 0, 142],      # car - dark blue
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ]
        
        for i, color in enumerate(colors):
            color_map[i] = color
        
        return color_map
    
    def segment(self, frame: np.ndarray) -> Dict:
        """
        Perform semantic segmentation on frame with frame-skip caching
        
        Args:
            frame: Input BGR image
            
        Returns:
            Dict containing:
                - segmentation_map: Class predictions (H x W)
                - road_mask: Binary road mask
                - parking_mask: Binary parking/sidewalk mask
                - obstacle_map: Map of obstacles
                - confidence: Segmentation confidence
                - colored_map: Visualization (BGR)
                - cached: Whether result is from cache
        """
        if self.model is None:
            return self._fallback_segment(frame)
        
        # Frame skip: Process every 3 frames, cache others
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0 and self.cached_result is not None:
            # Return cached result
            result = self.cached_result.copy()
            result['cached'] = True
            return result
        
        # Process new frame
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predictions
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original size
        segmentation_map = cv2.resize(pred.astype(np.uint8), (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Calculate confidence (use softmax probabilities)
        probs = torch.softmax(logits, dim=1)
        confidence_map = probs.max(dim=1)[0].squeeze(0).cpu().numpy()
        confidence_map = cv2.resize(confidence_map, (w, h))
        avg_confidence = float(np.mean(confidence_map))
        
        # Create masks for different categories
        road_mask = np.isin(segmentation_map, self.drivable_classes).astype(np.uint8) * 255
        sidewalk_mask = (segmentation_map == 1).astype(np.uint8) * 255  # Class 1 = sidewalk
        parking_mask = np.isin(segmentation_map, self.parking_classes).astype(np.uint8) * 255
        obstacle_map = np.isin(segmentation_map, self.obstacle_classes).astype(np.uint8) * 255
        
        # Erode sidewalk from road to create clear separation
        kernel = np.ones((5, 5), np.uint8)
        sidewalk_dilated = cv2.dilate(sidewalk_mask, kernel, iterations=2)
        road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(sidewalk_dilated))
        
        # Create colored visualization
        colored_map = self.color_map[segmentation_map]
        
        # Create result dict
        result = {
            'segmentation_map': segmentation_map,
            'road_mask': road_mask,
            'parking_mask': parking_mask,
            'obstacle_map': obstacle_map,
            'confidence': avg_confidence,
            'confidence_map': confidence_map,
            'colored_map': colored_map,
            'cached': False
        }
        
        # Cache this result for next frames
        self.cached_result = result.copy()
        
        return result
    
    def _fallback_segment(self, frame: np.ndarray) -> Dict:
        """Simple fallback segmentation without deep learning"""
        h, w = frame.shape[:2]
        
        # Simple color-based segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Road detection (dark, low saturation)
        lower = np.array([0, 0, 20])
        upper = np.array([180, 50, 150])
        road_mask = cv2.inRange(hsv, lower, upper)
        
        # Focus on bottom half
        mask_top = np.zeros_like(road_mask)
        mask_top[int(h*0.4):, :] = 255
        road_mask = cv2.bitwise_and(road_mask, mask_top)
        
        # Clean up
        kernel = np.ones((15, 15), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        # Take largest connected component
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            road_mask = np.zeros_like(road_mask)
            cv2.drawContours(road_mask, [largest], -1, 255, -1)
        
        # Simple obstacle detection
        _, obstacle_map = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        obstacle_map = cv2.bitwise_and(obstacle_map, cv2.bitwise_not(road_mask))
        
        # Parking (adjacent to road)
        parking_mask = cv2.dilate(road_mask, kernel, iterations=1)
        parking_mask = cv2.bitwise_xor(parking_mask, road_mask)
        
        # Colored map (simple)
        colored_map = frame.copy()
        colored_map[road_mask > 0] = cv2.addWeighted(
            colored_map[road_mask > 0], 0.7,
            np.array([128, 64, 128], dtype=np.uint8), 0.3, 0
        )
        
        return {
            'segmentation_map': np.zeros((h, w), dtype=np.uint8),
            'road_mask': road_mask,
            'parking_mask': parking_mask,
            'obstacle_map': obstacle_map,
            'confidence': 0.5,
            'confidence_map': np.ones((h, w)) * 0.5,
            'colored_map': colored_map
        }
    
    def get_road_boundaries(self, road_mask: np.ndarray, 
                           num_points: int = 30) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract left and right road boundaries from segmentation mask
        
        Args:
            road_mask: Binary road mask
            num_points: Number of points to sample
            
        Returns:
            (left_boundary, right_boundary) as Nx2 arrays
        """
        h, w = road_mask.shape
        
        left_points = []
        right_points = []
        
        # Sample at different heights
        y_start = int(h * 0.5)
        for y in range(y_start, h, max(1, (h - y_start) // num_points)):
            row = road_mask[y, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 5:
                left_edge = road_pixels[0]
                right_edge = road_pixels[-1]
                
                left_points.append([left_edge, y])
                right_points.append([right_edge, y])
        
        if len(left_points) < 5 or len(right_points) < 5:
            return None, None
        
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # Fit smooth curves
        try:
            z_left = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            y_vals = np.linspace(y_start, h, num_points)
            x_left = z_left[0] * y_vals**2 + z_left[1] * y_vals + z_left[2]
            left_boundary = np.column_stack([x_left, y_vals])
            
            z_right = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
            x_right = z_right[0] * y_vals**2 + z_right[1] * y_vals + z_right[2]
            right_boundary = np.column_stack([x_right, y_vals])
            
            return left_boundary, right_boundary
        except:
            return None, None
    
    def get_safe_path(self, road_mask: np.ndarray, 
                     obstacle_map: np.ndarray = None,
                     num_points: int = 30) -> Optional[np.ndarray]:
        """
        Calculate safe navigation path through road
        
        Args:
            road_mask: Binary road mask
            obstacle_map: Optional obstacle map to avoid
            num_points: Number of path points
            
        Returns:
            Safe path as Nx2 array of (x, y) coordinates
        """
        h, w = road_mask.shape
        
        # Combine road mask with obstacle avoidance
        if obstacle_map is not None:
            kernel = np.ones((15, 15), np.uint8)
            obstacles_dilated = cv2.dilate(obstacle_map, kernel, iterations=2)
            safe_road = cv2.bitwise_and(road_mask, cv2.bitwise_not(obstacles_dilated))
        else:
            safe_road = road_mask
        
        path_points = []
        
        # Find centerline
        y_start = int(h * 0.5)
        for y in range(y_start, h, max(1, (h - y_start) // num_points)):
            row = safe_road[y, :]
            road_pixels = np.where(row > 0)[0]
            
            if len(road_pixels) > 5:
                center_x = int(np.median(road_pixels))
                path_points.append([center_x, y])
        
        if len(path_points) < 5:
            return None
        
        path_points = np.array(path_points)
        
        # Smooth path
        try:
            z = np.polyfit(path_points[:, 1], path_points[:, 0], 2)
            y_vals = np.linspace(y_start, h, num_points)
            x_vals = z[0] * y_vals**2 + z[1] * y_vals + z[2]
            safe_path = np.column_stack([x_vals, y_vals])
            
            return safe_path
        except:
            return None
