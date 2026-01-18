"""
Moondream2 VLM Integration for ADAS System
Provides intelligent scene understanding and hazard detection
"""

import cv2
import time
import numpy as np
from PIL import Image
from collections import deque
import warnings

warnings.filterwarnings('ignore')


class Moondream2ADAS:
    """
    Moondream2 VLM wrapper for ADAS system
    Replaces rule-based detectors with intelligent scene understanding
    """
    
    # ADAS-specific query templates (varied, driving-focused)
    QUERIES = {
        'hazards': "Describe road hazards ahead: turns, intersections, bumps, obstacles, or vehicles. Be specific and concise.",
        'scene': "Describe the road condition and any important features for safe driving.",
        'instruction': "What driving action should be taken based on visible conditions?",
        'objects': "List vehicles and obstacles on the drivable road surface.",
    }
    
    def __init__(self, frame_skip=30, max_tokens=32, verbose=True):
        """
        Initialize Moondream2 for ADAS
        
        Args:
            frame_skip: Process VLM every N frames (30 = 1 sec at 30fps)
            max_tokens: Max tokens per response (32 = concise)
            verbose: Print initialization info
        """
        self.frame_skip = frame_skip
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.frame_count = 0
        
        # Cached results
        self.cached_hazards = None
        self.cached_scene = None
        self.cached_instruction = None
        self.cache_frame_num = -1
        
        # Performance tracking
        self.inference_times = deque(maxlen=10)
        
        # Import and initialize model
        try:
            if verbose:
                print("🔮 Initializing Moondream2 VLM...")
            
            from run_moondream_image import OptimizedMoondream
            self.model = OptimizedMoondream(verbose=verbose)
            
            if verbose:
                print("✓ Moondream2 VLM ready for ADAS")
                print(f"  Frame skip: Every {frame_skip} frames")
                print(f"  Max tokens: {max_tokens}")
                print(f"  Expected overhead: ~{300/frame_skip:.1f}ms per frame")
        
        except ImportError as e:
            if verbose:
                print("⚠️  Moondream2 not available")
                print("   Make sure run_moondream_image.py is in the same directory")
            self.model = None
    
    def is_available(self):
        """Check if VLM is loaded and available"""
        return self.model is not None
    
    def should_process_frame(self, frame_num):
        """Check if this frame should be processed by VLM"""
        return frame_num % self.frame_skip == 0
    
    def analyze_frame(self, frame, frame_num=None, tracks=None):
        """
        Analyze frame with Moondream2 VLM
        
        Args:
            frame: BGR image from OpenCV
            frame_num: Frame number for tracking
            tracks: List of Track objects (for velocity context)
            
        Returns:
            {
                'hazards': str,
                'scene': str,
                'instruction': str,
                'inference_time': float,
                'cached': bool
            }
        """
        if not self.is_available():
            return {
                'hazards': "VLM not available",
                'scene': "VLM not available",
                'instruction': "Continue driving normally",
                'inference_time': 0.0,
                'cached': True
            }
        
        self.frame_count += 1
        
        # Check if we should use cached results
        if frame_num is not None:
            if not self.should_process_frame(frame_num):
                return {
                    'hazards': self.cached_hazards or "No hazards detected",
                    'scene': self.cached_scene or "Road scene",
                    'instruction': self.cached_instruction or "Continue driving",
                    'inference_time': 0.0,
                    'cached': True
                }
        
        # Build context from tracking data
        context = ""
        if tracks:
            moving = sum(1 for t in tracks if hasattr(t, 'velocity') and abs(t.velocity) > 0.5)
            stationary = sum(1 for t in tracks if hasattr(t, 'velocity') and abs(t.velocity) < 0.2)
            if moving > 0 or stationary > 0:
                context = f"Context: {moving} moving vehicles, {stationary} stationary. "
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Run VLM inference
        t_start = time.time()
        
        # Query for hazards (most important for ADAS) with context
        hazard_query = context + self.QUERIES['hazards']
        hazard_result = self.model.query(
            pil_image,
            hazard_query,
            max_tokens=self.max_tokens,
            verbose=False
        )
        
        # Query for scene description
        scene_result = self.model.query(
            pil_image,
            self.QUERIES['scene'],
            max_tokens=self.max_tokens,
            verbose=False
        )
        
        # Query for driving instruction
        instruction_result = self.model.query(
            pil_image,
            self.QUERIES['instruction'],
            max_tokens=self.max_tokens,
            verbose=False
        )
        
        inference_time = time.time() - t_start
        self.inference_times.append(inference_time)
        
        # Cache results
        self.cached_hazards = hazard_result['answer']
        self.cached_scene = scene_result['answer']
        self.cached_instruction = instruction_result['answer']
        self.cache_frame_num = frame_num
        
        return {
            'hazards': self.cached_hazards,
            'scene': self.cached_scene,
            'instruction': self.cached_instruction,
            'inference_time': inference_time,
            'cached': False
        }
    
    def get_stats(self):
        """Get performance statistics"""
        if len(self.inference_times) == 0:
            return {
                'avg_inference_time': 0.0,
                'frames_processed': 0,
                'avg_overhead_per_frame': 0.0
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        overhead_per_frame = avg_time / self.frame_skip
        
        return {
            'avg_inference_time': avg_time,
            'frames_processed': self.frame_count,
            'avg_overhead_per_frame': overhead_per_frame,
            'cache_hit_rate': (self.frame_count - len(self.inference_times)) / max(1, self.frame_count)
        }
    
    def reset(self):
        """Reset cached results"""
        self.cached_hazards = None
        self.cached_scene = None
        self.cached_instruction = None
        self.cache_frame_num = -1
        self.frame_count = 0


class Moondream2Lite:
    """
    Lightweight wrapper for ADAS with single query per frame
    More efficient - uses only hazard detection query
    """
    
    def __init__(self, frame_skip=30, max_tokens=32, verbose=True):
        """Initialize lite version with single query"""
        self.frame_skip = frame_skip
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.frame_count = 0
        
        self.cached_result = "No hazards detected"
        self.inference_times = deque(maxlen=10)
        
        try:
            if verbose:
                print("🔮 Initializing Moondream2 VLM (Lite)...")
            
            from run_moondream_image import OptimizedMoondream
            self.model = OptimizedMoondream(verbose=verbose)
            
            if verbose:
                print("✓ Moondream2 VLM Lite ready")
                print(f"  Mode: Single query (hazards only)")
                print(f"  Frame skip: {frame_skip}")
                print(f"  Expected overhead: ~{100/frame_skip:.1f}ms per frame")
        
        except ImportError:
            if verbose:
                print("⚠️  Moondream2 not available")
            self.model = None
    
    def is_available(self):
        return self.model is not None
    
    def analyze_frame(self, frame, frame_num, tracks=None):
        """Analyze with single hazard detection query"""
        if not self.is_available():
            return {
                'hazards': "VLM not available",
                'inference_time': 0.0,
                'cached': True
            }
        
        self.frame_count += 1
        
        # Use cached result if not time to process
        if frame_num % self.frame_skip != 0:
            return {
                'hazards': self.cached_result,
                'inference_time': 0.0,
                'cached': True
            }
        
        # Build context from tracking data
        context = ""
        if tracks:
            moving = sum(1 for t in tracks if hasattr(t, 'velocity') and abs(t.velocity) > 0.5)
            stationary = sum(1 for t in tracks if hasattr(t, 'velocity') and abs(t.velocity) < 0.2)
            if moving > 0 or stationary > 0:
                context = f"Context: {moving} moving, {stationary} parked. "
        
        # Convert and process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        t_start = time.time()
        query = context + "Describe road hazards with location and distance: turns, bumps, intersections, obstacles, vehicles. Example: 'Silver sedan 4m ahead. Road curves right 20m ahead.'"
        result = self.model.query(
            pil_image,
            query,
            max_tokens=self.max_tokens,
            verbose=False
        )
        inference_time = time.time() - t_start
        
        self.cached_result = result['answer']
        self.inference_times.append(inference_time)
        
        return {
            'hazards': self.cached_result,
            'inference_time': inference_time,
            'cached': False
        }
    
    def get_stats(self):
        """Get stats"""
        if len(self.inference_times) == 0:
            return {'avg_inference_time': 0.0, 'avg_overhead': 0.0}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        return {
            'avg_inference_time': avg_time,
            'avg_overhead': avg_time / self.frame_skip
        }


# Example usage in ADAS system
if __name__ == "__main__":
    print("="*70)
    print("MOONDREAM2 VLM - ADAS INTEGRATION TEST")
    print("="*70)
    
    # Test full version
    vlm = Moondream2ADAS(frame_skip=30, max_tokens=32, verbose=True)
    
    if vlm.is_available():
        print("\n✓ VLM initialized successfully")
        print("\n📊 Configuration:")
        print(f"  Frame skip: {vlm.frame_skip} frames")
        print(f"  Max tokens: {vlm.max_tokens}")
        print(f"  Queries: {len(vlm.QUERIES)}")
        print("\n📋 Available queries:")
        for key, query in vlm.QUERIES.items():
            print(f"  - {key}: {query[:50]}...")
    else:
        print("\n❌ VLM not available")
    
    print("\n" + "="*70)
    print("Integration ready for ADAS system!")
    print("="*70)
