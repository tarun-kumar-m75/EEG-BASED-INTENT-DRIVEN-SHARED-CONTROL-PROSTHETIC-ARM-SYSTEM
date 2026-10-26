"""
src/preprocessing.py
====================
EEG Preprocessing Pipeline for Motor-Imagery and Intent Decoding.

Technical & Neurophysiological Rationale:
------------------------------------------
1. Common Average Referencing (CAR):
   - Formula: V_i_CAR(t) = V_i(t) - (1/N) * sum_{j=1}^N V_j(t)
   - Rationale: Scalp EEG electrodes record potential differences relative to a reference.
     CAR subtracts instantaneous global average activity, attenuating common-mode
     noise and distant environmental artifacts without distorting local sensorimotor dipoles.

2. Band-pass Filtering (8.0 Hz - 30.0 Hz):
   - Filter: 4th-order zero-phase Butterworth filter.
   - Rationale: Voluntary motor intention and motor imagery induce Event-Related
     Desynchronization (ERD) and Event-Related Synchronization (ERS) in the sensorimotor
     cortex. These phenomena are concentrated strictly within:
       * Mu rhythm: 8.0 - 13.0 Hz (sensory-motor rhythm over motor cortex)
       * Beta rhythm: 13.0 - 30.0 Hz (post-movement rebound and motor cortex activation)
     Frequencies below 8 Hz (ocular blinks, DC drift) and above 30 Hz (EMG cranial muscle
     contraction) are eliminated.

3. Notch Filtering (50.0 / 60.0 Hz):
   - Filter: 2nd-order IIR Notch filter (Q-factor = 30).
   - Rationale: Suppresses ubiquitous power grid electromagnetic interference.

4. Dynamic Mu and Beta Band Power:
   - Evaluated via Welch's Power Spectral Density (PSD) or FFT integration:
     P_mu = sum_{f=8}^{13} PSD(f) * df
     P_beta = sum_{f=13}^{30} PSD(f) * df
   - Calculated dynamically on motor-cortex electrodes (C3 and C4) to expose ERD/ERS.
"""

import numpy as np
from scipy import signal
import mne
from typing import Dict, Tuple, List, Optional

class EEGPreprocessor:
    def __init__(
        self,
        sfreq: float = 160.0,
        low_freq: float = 8.0,
        high_freq: float = 30.0,
        notch_freq: float = 50.0,
        filter_order: int = 4
    ):
        """
        Initializes EEG Preprocessor with filter parameters.
        """
        self.sfreq = sfreq
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        
        # Design zero-phase digital Butterworth bandpass filter using Second-Order Sections (SOS)
        # SOS format provides numerical stability across high-order and narrow-band filtering.
        self.bp_sos = signal.butter(
            N=filter_order,
            Wn=[low_freq, high_freq],
            btype='bandpass',
            output='sos',
            fs=sfreq
        )
        
        # Design IIR Notch filter
        nyq = sfreq / 2.0
        if notch_freq < nyq:
            self.notch_b, self.notch_a = signal.iirnotch(w0=notch_freq, Q=30.0, fs=sfreq)
            self.has_notch = True
        else:
            self.has_notch = False

    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Applies Common Average Referencing (CAR).
        Args:
            data: shape (n_channels, n_samples)
        Returns:
            car_data: shape (n_channels, n_samples)
        """
        # Average across all spatial channels at each time point
        mean_spatial = np.mean(data, axis=0, keepdims=True)
        return data - mean_spatial

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase 8-30 Hz Butterworth bandpass filter.
        Args:
            data: shape (n_channels, n_samples)
        Returns:
            filtered_data: shape (n_channels, n_samples)
        """
        # sosfiltfilt applies forward-backward zero-phase filtering
        return signal.sosfiltfilt(self.bp_sos, data, axis=-1)

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase IIR notch filter.
        Args:
            data: shape (n_channels, n_samples)
        Returns:
            notched_data: shape (n_channels, n_samples)
        """
        if not self.has_notch:
            return data
        return signal.filtfilt(self.notch_b, self.notch_a, data, axis=-1)

    def process_window(self, window_data: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline for an incoming sliding window of EEG.
        Steps: CAR Referencing -> Notch Filtering -> Mu/Beta Bandpass Filtering.
        Args:
            window_data: shape (n_channels, n_samples) in microvolts (uV)
        Returns:
            processed_data: shape (n_channels, n_samples)
        """
        # Step 1: Re-referencing (Common Average Reference)
        car_data = self.apply_car(window_data)
        
        # Step 2: Notch filter for power line interference
        notched_data = self.apply_notch(car_data)
        
        # Step 3: Mu/Beta bandpass filter (8-30 Hz)
        filtered_data = self.apply_bandpass(notched_data)
        
        return filtered_data

    def compute_band_powers(
        self,
        data: np.ndarray,
        ch_names: Optional[List[str]] = None,
        target_channels: Tuple[str, str] = ('C3', 'C4')
    ) -> Dict[str, float]:
        """
        Computes dynamic Mu (8-13 Hz) and Beta (13-30 Hz) band powers using Welch's PSD.
        Calculates contralateral/ipsilateral power values on sensorimotor electrodes.
        
        Returns:
            Dictionary with mu_power, beta_power, c3_mu, c4_mu, mu_lateralization_index.
        """
        n_samples = data.shape[1]
        nperseg = min(n_samples, int(self.sfreq * 1.0))  # 1.0s window for Welch
        if nperseg < 16:
            return {
                "mu_power": 0.0,
                "beta_power": 0.0,
                "c3_mu": 0.0,
                "c4_mu": 0.0,
                "c3_beta": 0.0,
                "c4_beta": 0.0,
                "lateralization_index": 0.0
            }
            
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=nperseg, axis=-1)
        
        # Frequency masks
        mu_mask = (freqs >= 8.0) & (freqs <= 13.0)
        beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
        
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        
        # Mean across all channels
        global_psd = np.mean(psd, axis=0)
        mu_power = float(np.sum(global_psd[mu_mask]) * df)
        beta_power = float(np.sum(global_psd[beta_mask]) * df)
        
        # Specific C3 and C4 channel values if available
        c3_mu, c4_mu = mu_power, mu_power
        c3_beta, c4_beta = beta_power, beta_power
        
        if ch_names is not None:
            clean_names = [ch.upper().replace('.', '') for ch in ch_names]
            if target_channels[0] in clean_names:
                idx_c3 = clean_names.index(target_channels[0])
                c3_mu = float(np.sum(psd[idx_c3, mu_mask]) * df)
                c3_beta = float(np.sum(psd[idx_c3, beta_mask]) * df)
            if target_channels[1] in clean_names:
                idx_c4 = clean_names.index(target_channels[1])
                c4_mu = float(np.sum(psd[idx_c4, mu_mask]) * df)
                c4_beta = float(np.sum(psd[idx_c4, beta_mask]) * df)
                
        # Lateralization index: (C4 - C3) / (C4 + C3)
        # Positive index indicates left-hemisphere desynchronization (right-side intent)
        denom = (c4_mu + c3_mu) if (c4_mu + c3_mu) > 1e-9 else 1.0
        lat_index = float((c4_mu - c3_mu) / denom)
        
        return {
            "mu_power": round(mu_power, 3),
            "beta_power": round(beta_power, 3),
            "c3_mu": round(c3_mu, 3),
            "c4_mu": round(c4_mu, 3),
            "c3_beta": round(c3_beta, 3),
            "c4_beta": round(c4_beta, 3),
            "lateralization_index": round(lat_index, 4)
        }

    def epoch_mne_raw(
        self,
        raw: mne.io.Raw,
        tmin: float = 0.5,
        tmax: float = 2.5,
        event_ids: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocesses and extracts epochs from continuous MNE raw object.
        Applies CAR, Notch, and Bandpass filtering, then extracts epochs.
        
        Args:
            raw: mne.io.Raw instance
            tmin, tmax: trial window bounds in seconds relative to cue
            event_ids: mapping of annotation description to integer label
            
        Returns:
            epochs_data: shape (n_epochs, n_channels, n_times) in uV
            labels: shape (n_epochs,)
            class_names: list of class names
        """
        if event_ids is None:
            event_ids = {'T0': 0, 'T1': 1, 'T2': 2}
            
        # Copy raw to prevent in-place mutation
        raw_proc = raw.copy()
        
        # Apply CAR referencing
        raw_proc.set_eeg_reference('average', projection=False, verbose=False)
        
        # Apply bandpass filter (8-30 Hz)
        raw_proc.filter(
            l_freq=self.low_freq,
            h_freq=self.high_freq,
            fir_design='firwin',
            skip_by_annotation='edge',
            verbose=False
        )
        
        # Extract events from annotations
        events, event_dict = mne.events_from_annotations(raw_proc, verbose=False)
        
        # Filter events of interest
        matched_dict = {k: v for k, v in event_dict.items() if k in event_ids}
        
        epochs = mne.Epochs(
            raw_proc,
            events,
            event_id=matched_dict,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )
        
        # Data in microvolts (uV)
        epochs_data = epochs.get_data() * 1e6
        
        # Remap event codes to standardized [0, 1, 2]
        inv_dict = {v: k for k, v in matched_dict.items()}
        mapped_labels = np.array([event_ids[inv_dict[code]] for code in epochs.events[:, 2]])
        
        return epochs_data, mapped_labels, list(event_ids.keys())

if __name__ == "__main__":
    print("[*] Running unit test on EEGPreprocessor...")
    preprocessor = EEGPreprocessor(sfreq=160.0, low_freq=8.0, high_freq=30.0)
    
    # Test on synthetic multi-channel window: 16 channels, 320 samples (2.0s at 160Hz)
    np.random.seed(42)
    sample_window = np.random.randn(16, 320) * 15.0  # 15 uV noise
    
    # Add 10 Hz mu oscillation to channel 6 (C3)
    t = np.arange(320) / 160.0
    sample_window[6, :] += 30.0 * np.sin(2 * np.pi * 10.0 * t)
    
    # Process window
    proc_window = preprocessor.process_window(sample_window)
    assert proc_window.shape == sample_window.shape, "Shape mismatch after processing"
    
    # Band power calculation
    ch_names = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2']
    powers = preprocessor.compute_band_powers(proc_window, ch_names=ch_names, target_channels=('C3', 'C4'))
    
    print("[OK] Preprocessed window shape:", proc_window.shape)
    print(f"[OK] Computed Mu Power (8-13 Hz): {powers['mu_power']} uV^2/Hz")
    print(f"[OK] Computed Beta Power (13-30 Hz): {powers['beta_power']} uV^2/Hz")
    print(f"[OK] C3 Mu: {powers['c3_mu']} uV^2/Hz | C4 Mu: {powers['c4_mu']} uV^2/Hz")
    print(f"[OK] Lateralization Index: {powers['lateralization_index']}")
    print("\n[OK] Phase 2: Preprocessing pipeline test completed successfully!")
