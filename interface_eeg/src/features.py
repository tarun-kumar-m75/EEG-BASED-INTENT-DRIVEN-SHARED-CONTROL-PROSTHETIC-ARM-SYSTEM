"""
src/features.py
===============
Common Spatial Pattern (CSP) and Spectral Feature Extraction Pipeline.

Neuroengineering Rationale:
---------------------------
Common Spatial Patterns (CSP) is the gold-standard spatial filtering algorithm
for motor-imagery Brain-Computer Interfaces (BCI).

Mathematical Formulation:
1. For class k with trial data X_k in R^{C x T}, the normalized spatial covariance is:
   Sigma_k = (X_k * X_k^T) / trace(X_k * X_k^T)
2. Average spatial covariances across trials: Sigma_1 and Sigma_2.
3. Composite covariance: Sigma_c = Sigma_1 + Sigma_2 = U * Lambda * U^T.
4. Whitening matrix: P = Lambda^{-1/2} * U^T.
5. Transformed covariances: S_1 = P * Sigma_1 * P^T and S_2 = P * Sigma_2 * P^T.
6. Generalized eigenvalue decomposition: S_1 * B = B * Lambda_1.
7. Projection matrix: W = B^T * P.
   - First m components maximize variance for class 1 (and minimize for class 2).
   - Last m components maximize variance for class 2 (and minimize for class 1).
8. Feature extraction:
   f_i = log( var(w_i^T * X) / sum_{j=1}^{2m} var(w_j^T * X) )
"""

import numpy as np
import mne
from mne.decoding import CSP
from typing import Tuple, Optional, List, Dict
import joblib

class EEGFeatureExtractor:
    def __init__(
        self,
        n_components: int = 4,
        log: bool = True,
        norm_trace: bool = True,
        cov_est: str = 'concat',
        transform_into: str = 'average_power'
    ):
        """
        Initializes CSP feature extractor.
        
        Args:
            n_components: Number of spatial filters per class (e.g. 4 or 6).
            log: If True, features are log-variance transformed.
            norm_trace: Normalize covariance matrix trace to mitigate artifact scaling.
            cov_est: Covariance estimator ('concat' or 'epoch').
        """
        self.n_components = n_components
        self.log = log
        self.norm_trace = norm_trace
        self.cov_est = cov_est
        
        # MNE CSP handles 2-class and multiclass (OVR / AJD)
        self.csp = CSP(
            n_components=n_components,
            log=log,
            norm_trace=norm_trace,
            cov_est=cov_est,
            transform_into=transform_into
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EEGFeatureExtractor':
        """
        Fits CSP spatial filters on training epochs.
        
        Args:
            X: Array of shape (n_epochs, n_channels, n_times) in Volts or uV
            y: Array of shape (n_epochs,) containing class integer labels
        """
        # Ensure float64 for high numerical precision in eigenvalue decomposition
        X_dbl = X.astype(np.float64)
        self.csp.fit(X_dbl, y)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extracts CSP log-variance features from preprocessed epochs.
        
        Args:
            X: Array of shape (n_epochs, n_channels, n_times)
               or single window of shape (n_channels, n_times)
        Returns:
            Features array of shape (n_epochs, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("CSP feature extractor must be fitted before transform().")
            
        # Handle single window input of shape (n_channels, n_times)
        is_single = (X.ndim == 2)
        if is_single:
            X_batch = np.expand_dims(X, axis=0)
        else:
            X_batch = X
            
        X_dbl = X_batch.astype(np.float64)
        features = self.csp.transform(X_dbl)
        
        if is_single:
            return features[0]
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fits CSP and transforms data in a single step."""
        return self.fit(X, y).transform(X)

    def get_spatial_patterns(self) -> np.ndarray:
        """Returns the learned spatial patterns (mixing matrix W^-1)."""
        if not self.is_fitted:
            raise RuntimeError("Extractor not fitted.")
        return self.csp.patterns_

    def get_spatial_filters(self) -> np.ndarray:
        """Returns the spatial filter projection matrix W."""
        if not self.is_fitted:
            raise RuntimeError("Extractor not fitted.")
        return self.csp.filters_

    def save(self, filepath: str):
        """Serializes the fitted CSP extractor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'EEGFeatureExtractor':
        """Loads a serialized CSP feature extractor from disk."""
        return joblib.load(filepath)

if __name__ == "__main__":
    print("[*] Running unit test on EEGFeatureExtractor (CSP)...")
    # Generate synthetic 2-class motor imagery trial data: 40 trials, 16 channels, 320 samples
    np.random.seed(42)
    n_epochs = 40
    n_channels = 16
    n_times = 320
    
    # Class 0: Strong activity on left hemisphere (e.g. channel 6 = C3)
    # Class 1: Strong activity on right hemisphere (e.g. channel 10 = C4)
    X = np.random.randn(n_epochs, n_channels, n_times) * 5.0
    y = np.array([0] * (n_epochs // 2) + [1] * (n_epochs // 2))
    
    t = np.arange(n_times) / 160.0
    for i in range(n_epochs):
        if y[i] == 0:
            X[i, 6, :] += 25.0 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz mu on C3
        else:
            X[i, 10, :] += 25.0 * np.sin(2 * np.pi * 10.0 * t) # 10 Hz mu on C4
            
    extractor = EEGFeatureExtractor(n_components=4)
    features = extractor.fit_transform(X, y)
    
    print("[OK] CSP fitted successfully!")
    print(f"[OK] Extracted features shape: {features.shape} (Expected: {n_epochs}, 4)")
    print(f"[OK] Sample feature vector (Trial 0, Class 0): {np.round(features[0], 3)}")
    print(f"[OK] Sample feature vector (Trial {n_epochs//2}, Class 1): {np.round(features[n_epochs//2], 3)}")
    
    # Test single-window streaming transform
    test_win = X[0]
    single_feat = extractor.transform(test_win)
    assert np.allclose(single_feat, features[0]), "Single-window transform mismatch"
    print("[OK] Single-window streaming transform validated!")
    print("\n[OK] Phase 3: Feature extraction unit test completed successfully!")
