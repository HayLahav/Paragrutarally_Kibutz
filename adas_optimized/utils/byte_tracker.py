"""Simplified ByteTrack for ADAS"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Track:
    id: int
    bbox: List[float]
    score: float
    class_id: int
    class_name: str = ""
    age: int = 0
    hits: int = 1

class ByteTracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8, max_age=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections):
        self.frame_count += 1
        dets_high = [d for d in detections if d['score'] >= self.track_thresh]
        dets_low = [d for d in detections if 0.1 <= d['score'] < self.track_thresh]
        
        matched, unmatched_tracks, unmatched_dets = self._match_detections(self.tracks, dets_high)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].bbox = dets_high[det_idx]['bbox']
            self.tracks[track_idx].score = dets_high[det_idx]['score']
            self.tracks[track_idx].age = 0
            self.tracks[track_idx].hits += 1
            if 'class_name' in dets_high[det_idx]:
                self.tracks[track_idx].class_name = dets_high[det_idx]['class_name']
        
        # Try to match unmatched tracks with low confidence detections
        if len(dets_low) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracks_list = [self.tracks[i] for i in unmatched_tracks]
            matched_low, _, _ = self._match_detections(unmatched_tracks_list, dets_low, 0.5)
            for track_idx, det_idx in matched_low:
                if track_idx < len(unmatched_tracks):
                    orig_idx = unmatched_tracks[track_idx]
                    self.tracks[orig_idx].bbox = dets_low[det_idx]['bbox']
                    self.tracks[orig_idx].score = dets_low[det_idx]['score']
                    self.tracks[orig_idx].age = 0
                    if 'class_name' in dets_low[det_idx]:
                        self.tracks[orig_idx].class_name = dets_low[det_idx]['class_name']
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if dets_high[det_idx]['score'] >= self.track_thresh:
                new_track = Track(
                    id=self.next_id,
                    bbox=dets_high[det_idx]['bbox'],
                    score=dets_high[det_idx]['score'],
                    class_id=dets_high[det_idx]['class_id'],
                    class_name=dets_high[det_idx].get('class_name', 'unknown')
                )
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Age unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks
    
    def _match_detections(self, tracks, detections, thresh=None):
        if thresh is None: 
            thresh = self.match_thresh
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._iou(track.bbox, det['bbox'])
        
        matched, unmatched_tracks, unmatched_dets = [], list(range(len(tracks))), list(range(len(detections)))
        while unmatched_tracks and unmatched_dets:
            max_iou, best_t, best_d = 0, None, None
            for t in unmatched_tracks:
                for d in unmatched_dets:
                    if iou_matrix[t, d] > max_iou:
                        max_iou, best_t, best_d = iou_matrix[t, d], t, d
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
