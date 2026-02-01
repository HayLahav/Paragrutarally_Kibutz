"""Simplified ByteTrack with Kalman Filter for ADAS"""
import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

class KalmanBoxTracker:
    """
    Kalman Filter for bounding box tracking.
    State: [x_center, y_center, width, height, vx, vy, vw, vh]
    Measurement: [x_center, y_center, width, height]
    """
    count = 0
    def __init__(self, bbox):
        # Initialize Kalman Filter
        # dynamParams=8 (x, y, w, h, vx, vy, vw, vh)
        # measureParams=4 (x, y, w, h)
        self.kf = cv2.KalmanFilter(8, 4)
        
        # Transition Matrix (F)
        # x = x + vx*dt, y = y + vy*dt, ...
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1.0
            
        # Measurement Matrix (H)
        # We measure x, y, w, h
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        
        # Process Noise Covariance (Q)
        # Model uncertainty
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 0.5 # Assume velocity changes slowly
        
        # Measurement Noise Covariance (R)
        # Measurement uncertainty
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1 # Trust detection reasonably well
        
        # Error Covariance (P)
        self.kf.errorCovPre = np.eye(8, dtype=np.float32) 
        
        # Initialize state with first detection
        self.kf.statePost = self.convert_bbox_to_z(bbox)
        
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        self.history = []
        self.last_bbox = bbox
        
        self.score = 0.0
        self.class_id = -1
        self.class_name = ""

    def update(self, bbox, score=None, class_id=None, class_name=None):
        """Update state with new measurement"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.age = 0
        
        if score is not None: self.score = score
        if class_id is not None: self.class_id = class_id
        if class_name is not None: self.class_name = class_name
        
        self.last_bbox = bbox
        measurement = self.convert_bbox_to_z(bbox)
        self.kf.correct(measurement)

    def predict(self):
        """Predict next state"""
        # If area is too small, just use previous state
        if self.kf.statePost[2] + self.kf.statePost[6] <= 0:
            self.kf.statePost[2] *= 0.0
            self.kf.statePost[6] *= 0.0
            
        prediction = self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Return predicted bbox
        self.history.append(self.convert_x_to_bbox(prediction))
        return self.history[-1]

    def get_state(self):
        """Returns the current bounding box estimate"""
        return self.convert_x_to_bbox(self.kf.statePost)

    def get_velocity(self):
        """Returns the current velocity estimate [vx, vy]"""
        return [float(self.kf.statePost[4, 0]), float(self.kf.statePost[5, 0])]

    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([[x], [y], [w], [h]], dtype=np.float32)

    @staticmethod
    def convert_x_to_bbox(x):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = x[0], x[1], x[2], x[3]
        return [cx - w/2., cy - h/2., cx + w/2., cy + h/2.]


class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_age=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
        # Reset ID counter if needed, though usually handled by instance
        # KalmanBoxTracker.count = 0 
    
    def update(self, detections):
        self.frame_count += 1
        
        # separate detections by score
        dets_high = [d for d in detections if d['score'] >= self.track_thresh]
        dets_low = [d for d in detections if 0.1 <= d['score'] < self.track_thresh]
        
        # Predict new locations of existing tracks
        for t in self.tracks:
            t.predict()
            
        # Match high confidence detections
        # Use predicted state for matching
        track_candidates = [t for t in self.tracks]
        
        # Helper to wrap track objects for the matching function which expects objects with .bbox attribute
        # We dynamically attach .bbox to the track object based on its current state for matching
        for t in track_candidates:
            # t.bbox is not a native attribute, we set it for the _match_detections function
            # We use the predicted state from get_state() which uses statePost if updated, or prediction
            # But wait, we just called predict(), so we should use the prediction result or get_state?
            # get_state returns statePost. predict() updates statePre and statePost (if no correction).
            # We want the PREDICTED position for matching.
            # predict() returns the predicted bbox.
            t.bbox = t.history[-1] 
            
        matched, unmatched_tracks, unmatched_dets = self._match_detections(track_candidates, dets_high)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = dets_high[det_idx]
            track.update(det['bbox'], det['score'], det['class_id'], det.get('class_name', ''))
            
        # Match unmatched tracks with low confidence detections
        if len(dets_low) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracks_list = [self.tracks[i] for i in unmatched_tracks]
            matched_low, _, _ = self._match_detections(unmatched_tracks_list, dets_low, 0.5)
            
            for track_idx, det_idx in matched_low:
                # matched_low returns indices relative to unmatched_tracks_list
                real_track_idx = unmatched_tracks[track_idx]
                track = self.tracks[real_track_idx]
                det = dets_low[det_idx]
                track.update(det['bbox'], det['score'], det['class_id'], det.get('class_name', ''))
                
                # Remove from unmatched list so it's not removed later
                # Use a second pass list or set to track what is truly unmatched
        
        # We need to reconstruct the list of tracks
        # Actually, simpler: mark tracks as updated.
        
        # Let's clean up the logic:
        # 1. Match High
        # 2. Match Low (only for tracks unmatched in 1)
        # 3. Create New (only for high dets unmatched in 1)
        # 4. Remove Old
        
        # To do this cleanly with indices, let's keep sets
        
        # Re-run matching cleanly
        matches_high, unmatched_tracks_idx, unmatched_dets_high_idx = self._match_detections(self.tracks, dets_high)
        
        # Update high matches
        for t_idx, d_idx in matches_high:
            self.tracks[t_idx].update(
                dets_high[d_idx]['bbox'], 
                dets_high[d_idx]['score'], 
                dets_high[d_idx]['class_id'],
                dets_high[d_idx].get('class_name', '')
            )
            
        # Prepare for low matching
        tracks_for_low = [self.tracks[i] for i in unmatched_tracks_idx]
        matches_low, unmatched_tracks_low_idx, _ = self._match_detections(tracks_for_low, dets_low, 0.5)
        
        # Update low matches
        for t_local_idx, d_local_idx in matches_low:
            # Map back to original track
            t_orig_idx = unmatched_tracks_idx[t_local_idx]
            self.tracks[t_orig_idx].update(
                dets_low[d_local_idx]['bbox'],
                dets_low[d_local_idx]['score'],
                dets_low[d_local_idx]['class_id'],
                dets_low[d_local_idx].get('class_name', '')
            )
            
        # Create new tracks (from unmatched high dets)
        for d_idx in unmatched_dets_high_idx:
            new_track = KalmanBoxTracker(dets_high[d_idx]['bbox'])
            new_track.score = dets_high[d_idx]['score']
            new_track.class_id = dets_high[d_idx]['class_id']
            new_track.class_name = dets_high[d_idx].get('class_name', '')
            self.tracks.append(new_track)
            
        # Remove dead tracks
        # criteria: max_age
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Prepare output format
        # The main code expects objects with .id, .bbox, .class_name attributes
        # Our KalmanBoxTracker has these (bbox is accessed via property or method?)
        # We need to ensure .bbox attribute exists and is the current best estimate
        for t in self.tracks:
            # Update the bbox attribute to be the current state (either updated or predicted)
            # Use flatten() to ensure it's a simple list/array
            state_bbox = t.get_state()
            # Ensure it's a list of floats
            t.bbox = [float(x) for x in state_bbox]
            
        return self.tracks
    
    def _match_detections(self, tracks, detections, thresh=None):
        if thresh is None: 
            thresh = self.match_thresh
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                # track.bbox is set in the update loop before calling this
                iou_matrix[t, d] = self._iou(track.bbox, det['bbox'])
        
        matched, unmatched_tracks, unmatched_dets = [], list(range(len(tracks))), list(range(len(detections)))
        
        # Linear assignment (Greedy)
        # For better results we could use scipy.optimize.linear_sum_assignment (Hungarian)
        # but sticking to greedy for consistency with original code unless scipy is guaranteed.
        # Original code used greedy.
        
        if iou_matrix.size > 0:
            # Loop through highest IoUs
            # Flatten and sort args
            # Actually, standard greedy is fine
            pass
            
        # Greedy matching
        while unmatched_tracks and unmatched_dets:
            max_iou = 0
            best_t, best_d = -1, -1
            
            # Find global max IoU
            # Optimization: check if we can do this faster?
            # Just simple loop is fine for N < 100
            for t in unmatched_tracks:
                for d in unmatched_dets:
                    if iou_matrix[t, d] > max_iou:
                        max_iou = iou_matrix[t, d]
                        best_t = t
                        best_d = d
            
            if max_iou < thresh:
                break
                
            matched.append((best_t, best_d))
            unmatched_tracks.remove(best_t)
            unmatched_dets.remove(best_d)
            
        return matched, unmatched_tracks, unmatched_dets
    
    def _iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min, yi_min = max(x1_min, x2_min), max(y1_min, y2_min)
        xi_max, yi_max = min(x1_max, x2_max), min(y1_max, y2_max)
        inter = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        union = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - inter
        return inter / union if union > 0 else 0.0
