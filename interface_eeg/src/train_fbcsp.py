"""
src/train_fbcsp.py
==================
Trains and exports the production deployment pipeline for:
Filter-Bank Common Spatial Patterns (FBCSP) + Shrinkage LDA.

Uses the exact configuration validated from the user's notebook:
- Channels: 14 motor-strip electrodes
- Sub-bands: (8, 12), (12, 16), (16, 20), (20, 26), (26, 30) Hz
- Classifier: StandardScaler + LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
- Classes: 0: IDLE, 1: GRASP, 2: RELEASE
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.classifier import FilterBankCSPDeploymentPipeline, MOTOR_14_CHANNELS

mne.set_log_level("ERROR")

# Configuration matching notebook
DATA_PATH = "mne_data"
SUBJECT = 1
RUNS = [3, 4, 7, 8]
MOTOR_CHANNELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
]
CLEAN_MOTOR_CHANNELS = [ch.replace(".", "").upper() for ch in MOTOR_CHANNELS]
FREQ_BANDS = [(8, 12), (12, 16), (16, 20), (20, 26), (26, 30)]
TMIN, TMAX = 0.0, 4.0


# Uses FilterBankCSPDeploymentPipeline imported from src.classifier


def load_and_epoch_training_data():
    raws = []
    for run in RUNS:
        paths = eegbci.load_data(SUBJECT, [run], path=DATA_PATH, update_path=True, verbose=False)
        raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
        raw.pick([ch for ch in MOTOR_CHANNELS if ch in raw.ch_names])
        raws.append(raw)
        
    def make_epochs(l_freq, h_freq):
        all_epochs = []
        for raw in raws:
            filtered = raw.copy()
            filtered.filter(l_freq, h_freq, method="fir", verbose=False)
            filtered.notch_filter(50, verbose=False)
            filtered.set_eeg_reference("average", projection=True, verbose=False)
            filtered.apply_proj(verbose=False)
            events, event_id = mne.events_from_annotations(filtered, verbose=False)
            all_epochs.append(mne.Epochs(
                filtered, events,
                event_id={"idle": event_id["T0"], "grasp": event_id["T1"], "release": event_id["T2"]},
                tmin=TMIN, tmax=TMAX, baseline=None, preload=True,
                reject=dict(eeg=200e-6), verbose=False,
            ))
        return mne.concatenate_epochs(all_epochs)

    print("[*] Filtering sub-bands and generating training epochs...")
    X_bands = []
    y = None
    for low, high in FREQ_BANDS:
        ep = make_epochs(low, high)
        X_bands.append(ep.get_data())
        if y is None:
            y = LabelEncoder().fit_transform(ep.events[:, 2])
            
    return X_bands, y


def build_and_save_deployment_model(output_path="models/fbcsp_deployment_pipeline.joblib"):
    os.makedirs("models", exist_ok=True)
    X_bands, y = load_and_epoch_training_data()
    
    print(f"[*] Training FBCSP + Shrinkage LDA on {len(y)} epochs...")
    pipeline = FilterBankCSPDeploymentPipeline(
        freq_bands=FREQ_BANDS,
        sfreq=160.0,
        channel_names=CLEAN_MOTOR_CHANNELS,
        class_mapping={0: "REST", 1: "GRASP", 2: "RELEASE"}
    )
    pipeline.fit(X_bands, y)
    
    joblib.dump(pipeline, output_path)
    print(f"[OK] Saved FBCSP deployment pipeline to: {output_path}")
    
    # Test on single window
    test_win = X_bands[0][0] # 14 channels, samples
    res = pipeline.predict_window(test_win)
    print(f"[OK] Verification Window Test: {res['predicted_class']} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"     Probabilities: {res['probabilities']}")


if __name__ == "__main__":
    build_and_save_deployment_model()
