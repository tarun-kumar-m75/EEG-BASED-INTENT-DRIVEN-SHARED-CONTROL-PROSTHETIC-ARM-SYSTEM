"""
src/replay.py
=============
Real-Time EEG Dataset Replay Engine.

Technical Details:
------------------
Simulates a real-time streaming acquisition device (e.g. Emotiv or OpenBCI)
by progressively streaming sliding windows of pre-recorded PhysioNet EEG data.
Tracks ground-truth event transitions, active window bounds, and cursor pointer.
"""

import os
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any

class EEGDatasetReplayStream:
    def __init__(
        self,
        data_path: str = "data/test_replay_stream.npz",
        window_duration_sec: float = 2.0,
        step_duration_sec: float = 0.25,
        context_duration_sec: float = 6.0
    ):
        self.data_path = data_path
        self.window_duration_sec = window_duration_sec
        self.step_duration_sec = step_duration_sec
        self.context_duration_sec = context_duration_sec
        
        # Load dataset
        if os.path.exists(data_path):
            archive = np.load(data_path, allow_pickle=True)
            self.raw_data = archive['data']            # Shape: (n_channels, n_samples)
            self.channel_names = list(archive['channels'])
            self.sfreq = float(archive['sfreq'])
            self.timestamps = archive['timestamps']
            self.ground_truth = archive['ground_truth']
            self.ground_truth_labels = list(archive['ground_truth_labels'])
            self.is_synthetic = False
        else:
            print(f"[!] Warning: {data_path} not found. Initializing synthetic benchmark stream.")
            self.sfreq = 160.0
            self.channel_names = [f"CH{i+1}" for i in range(16)]
            n_samples = int(self.sfreq * 120.0)
            self.raw_data = np.random.randn(16, n_samples).astype(np.float32) * 15.0
            self.timestamps = np.arange(n_samples) / self.sfreq
            self.ground_truth = np.zeros(n_samples, dtype=np.int32)
            self.ground_truth_labels = ['REST'] * n_samples
            self.is_synthetic = True
            
        self.n_channels, self.total_samples = self.raw_data.shape
        self.total_duration_sec = float(self.total_samples / self.sfreq)
        
        self.window_samples = int(self.window_duration_sec * self.sfreq)
        self.step_samples = int(self.step_duration_sec * self.sfreq)
        self.context_samples = int(self.context_duration_sec * self.sfreq)
        
        self.current_sample_idx = 0
        self.stream_source_description = "DATASET REPLAY: PhysioNet EEGBCI (S001)"

    def reset(self):
        """Resets stream back to time 0.0s."""
        self.current_sample_idx = 0

    def seek_time(self, time_sec: float):
        """Seeks to a specific timestamp in the recording."""
        target_sample = int(time_sec * self.sfreq)
        self.current_sample_idx = max(0, min(self.total_samples - self.window_samples, target_sample))

    def has_more_data(self) -> bool:
        return (self.current_sample_idx + self.window_samples) <= self.total_samples

    def get_next_window(self) -> Optional[Dict[str, Any]]:
        """
        Advances the stream by step_samples and returns the latest sliding window,
        along with the surrounding display context and pointer timestamps.
        """
        start_idx = self.current_sample_idx
        end_idx = start_idx + self.window_samples
        
        if end_idx > self.total_samples:
            # Loop replay continuously for endless demonstration
            self.current_sample_idx = 0
            start_idx = 0
            end_idx = self.window_samples
            
        # Active processing window (passed to ML classifier)
        window_data = self.raw_data[:, start_idx:end_idx]
        
        win_start_time = float(start_idx / self.sfreq)
        win_end_time = float(end_idx / self.sfreq)
        current_pointer_time = win_end_time  # Current cursor at end of window
        
        # Surrounding context buffer for visualization (e.g. 6 seconds total: 4s past + 2s current)
        ctx_start_idx = max(0, end_idx - self.context_samples)
        context_data = self.raw_data[:, ctx_start_idx:end_idx]
        context_time_axis = np.linspace(ctx_start_idx / self.sfreq, win_end_time, context_data.shape[1])
        
        # Ground truth during this window
        mid_idx = start_idx + (self.window_samples // 2)
        gt_code = int(self.ground_truth[mid_idx])
        gt_label = str(self.ground_truth_labels[mid_idx])
        
        # Advance pointer for next step
        self.current_sample_idx += self.step_samples
        progress_pct = (self.current_sample_idx / self.total_samples) * 100.0
        
        return {
            "window_data": window_data,               # Shape: (n_channels, window_samples)
            "win_start_time": round(win_start_time, 2),
            "win_end_time": round(win_end_time, 2),
            "current_pointer_time": round(current_pointer_time, 2),
            "current_time_sec": round(current_pointer_time, 2),
            "context_data": context_data,             # Shape: (n_channels, context_samples)
            "context_time_axis": context_time_axis,
            "progress_pct": round(progress_pct, 1),
            "ground_truth_code": gt_code,
            "ground_truth_label": gt_label,           # "REST", "GRASP", "RELEASE"
            "sfreq": self.sfreq,
            "n_channels": self.n_channels,
            "channel_names": self.channel_names,
            "source_flag": "EEG SOURCE: DATASET REPLAY"
        }

if __name__ == "__main__":
    stream = EEGDatasetReplayStream()
    sample = stream.get_next_window()
    print("[OK] Replay Stream initialized. Window:", sample["win_start_time"], "->", sample["win_end_time"])
    print("[OK] Pointer at:", sample["current_pointer_time"], "| Context shape:", sample["context_data"].shape)
