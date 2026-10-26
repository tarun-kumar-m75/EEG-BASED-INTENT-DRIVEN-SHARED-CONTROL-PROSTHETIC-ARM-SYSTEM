"""
src/classifier.py
=================
Regularized Linear Discriminant Analysis (Shrinkage LDA) and
Filter-Bank Common Spatial Patterns (FBCSP) Deployment Pipeline.
"""

import time
import os
import sys
from pathlib import Path
import json
import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import joblib

# Ensure root directory in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.preprocessing import EEGPreprocessor
    from src.features import EEGFeatureExtractor
except ImportError:
    from preprocessing import EEGPreprocessor
    from features import EEGFeatureExtractor

MOTOR_14_CHANNELS = [
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6"
]

class FilterBankCSPDeploymentPipeline:
    """
    Production-ready deployment class for online streaming sliding-window inference.
    Executes multi-band filtering -> CSP spatial projection -> Scaler -> Shrinkage LDA posteriors.
    """
    def __init__(
        self,
        freq_bands=None,
        sfreq: float = 160.0,
        channel_names: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, str]] = None
    ):
        self.freq_bands = freq_bands or [(8, 12), (12, 16), (16, 20), (20, 26), (26, 30)]
        self.sfreq = sfreq
        self.channel_names = channel_names or MOTOR_14_CHANNELS
        self.class_mapping = class_mapping or {0: "REST", 1: "GRASP", 2: "RELEASE"}
        self.csp_filters = []
        self.scaler = None
        self.lda = None
        
        # Pre-design SOS bandpass filters for zero-phase real-time filtering
        self.band_sos = []
        for low, high in self.freq_bands:
            sos = signal.butter(4, [low, high], btype='bandpass', output='sos', fs=self.sfreq)
            self.band_sos.append(sos)

    def fit(self, X_bands: List[np.ndarray], y: np.ndarray) -> 'FilterBankCSPDeploymentPipeline':
        """Fits CSP for each sub-band and trains shrinkage LDA."""
        from mne.decoding import CSP
        self.csp_filters = []
        train_feats = []
        
        for i, X_band in enumerate(X_bands):
            csp = CSP(n_components=4, reg="oas", log=True, norm_trace=False)
            feats = csp.fit_transform(X_band, y)
            self.csp_filters.append(csp)
            train_feats.append(feats)
            
        X_all = np.concatenate(train_feats, axis=1)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_all)
        
        self.lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
        self.lda.fit(X_scaled, y)
        return self

    def predict_window(self, window_data_14ch: np.ndarray) -> Dict[str, Any]:
        """
        Fast online streaming inference for a single window of shape (14, n_samples).
        Returns: predicted_class, confidence, probabilities dict, feature_vector.
        """
        feats_list = []
        for i, (low, high) in enumerate(self.freq_bands):
            filtered = signal.sosfiltfilt(self.band_sos[i], window_data_14ch, axis=-1)
            f = self.csp_filters[i].transform(np.expand_dims(filtered, axis=0))
            feats_list.append(f)
            
        all_feats = np.concatenate(feats_list, axis=1)
        scaled_feats = self.scaler.transform(all_feats)
        
        probas = self.lda.predict_proba(scaled_feats)[0]
        pred_label = int(self.lda.predict(scaled_feats)[0])
        pred_class = self.class_mapping.get(pred_label, str(pred_label))
        
        prob_dict = {
            self.class_mapping[0]: float(probas[0]),
            self.class_mapping[1]: float(probas[1]),
            self.class_mapping[2]: float(probas[2])
        }
        confidence = float(np.max(probas))
        
        return {
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "feature_vector": all_feats[0]
        }


class EEGIntentClassifier:
    def __init__(
        self,
        solver: str = 'lsqr',
        shrinkage: str = 'auto',
        class_names: Optional[List[str]] = None
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.class_names = class_names or ["REST", "GRASP", "RELEASE"]
        self.clf = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        return self.clf.predict_proba(X)


class EEGIntentDecoderPipeline:
    def __init__(
        self,
        preprocessor: Optional[EEGPreprocessor] = None,
        feature_extractor: Optional[EEGFeatureExtractor] = None,
        classifier: Optional[EEGIntentClassifier] = None,
        channel_names: Optional[List[str]] = None,
        class_mapping: Optional[Dict[int, str]] = None
    ):
        self.preprocessor = preprocessor or EEGPreprocessor(sfreq=160.0)
        self.feature_extractor = feature_extractor or EEGFeatureExtractor(n_components=4)
        self.classifier = classifier or EEGIntentClassifier()
        self.channel_names = channel_names or []
        self.class_mapping = class_mapping or {0: "REST", 1: "GRASP", 2: "RELEASE"}

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        self.feature_extractor.save(os.path.join(model_dir, "csp_extractor.joblib"))
        self.classifier.save(os.path.join(model_dir, "lda_classifier.joblib"))
        metadata = {
            "channel_names": self.channel_names,
            "class_mapping": self.class_mapping,
            "sfreq": self.preprocessor.sfreq,
            "low_freq": self.preprocessor.low_freq,
            "high_freq": self.preprocessor.high_freq,
            "notch_freq": self.preprocessor.notch_freq,
            "csp_components": self.feature_extractor.n_components
        }
        with open(os.path.join(model_dir, "pipeline_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
