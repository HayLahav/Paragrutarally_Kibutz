"""
Single Camera Distance Estimation
Uses known object sizes
"""

class DistanceEstimator:
    """Estimate distance to objects using monocular vision"""
    
    # Real-world object sizes in cm
    OBJECT_SIZES = {
        'person':    (170, 50),
        'bicycle':   (100, 65),
        'car':       (150, 180),
        'motorcycle':(100, 100),
        'bus':       (320, 250),
        'truck':     (350, 250),
    }
    
    def __init__(self, focal_length: float = 700.0):
        """
        Args:
            focal_length: Camera focal length in pixels (calibrate for your camera)
        """
        self.focal_length = focal_length
    
    def estimate(self, bbox, class_name):
        """
        Estimate distance to object
        
        Args:
            bbox: [x1, y1, x2, y2] in pixels
            class_name: Object class name
        
        Returns:
            Distance in meters (-1 if cannot estimate)
        """
        if class_name not in self.OBJECT_SIZES:
            return -1.0
        
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1
        
        if pixel_height <= 0:
            return -1.0
        
        # Get real-world height in cm
        real_height_cm, _ = self.OBJECT_SIZES[class_name]
        
        # Distance formula: D = (real_height * focal_length) / pixel_height
        # Convert cm to meters
        distance_m = (real_height_cm * self.focal_length) / (pixel_height * 100)
        
        return max(0.0, distance_m)
