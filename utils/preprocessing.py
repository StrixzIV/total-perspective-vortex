import mne
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pywt

def load_and_preprocess(subject, runs):
    """
    Load EEG data from PhysioNet, concatenate, filter, epoch, and return data.
    
    Parameters:
    subject (int): Subject number (1-109)
    runs (list of int): List of run numbers to load
    
    Returns:
    X (np.ndarray): Epochs data of shape (n_epochs, n_channels, n_times)
    y (np.ndarray): Labels mapped to 1 (T1) and 2 (T2)
    """
    # 1. Fetch .edf files
    edf_files = mne.datasets.eegbci.load_data(subject, runs, verbose='WARNING')
    
    # 2. Read and concatenate raws
    raws = [mne.io.read_raw_edf(f, preload=True, verbose='WARNING') for f in edf_files]
    raw = mne.concatenate_raws(raws)
    
    # 3. Clean channel names and set standard montage
    mne.datasets.eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    
    # 4. Bandpass filter to mu + beta bands (8-30 Hz)
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose='WARNING')
    
    # 5. Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose='WARNING')
    
    # 6. Map T1/T2 annotations to specific keys
    mapping = {}
    for key, val in event_id.items():
        if 'T1' in key:
            mapping['T1'] = val
        elif 'T2' in key:
            mapping['T2'] = val
            
    if 'T1' not in mapping or 'T2' not in mapping:
        raise ValueError(
            f"Could not find T1 and/or T2 event annotations in the loaded data for subject {subject}, "
            f"runs {runs}. Found: {event_id}"
        )
        
    # 7. Epoch data and drop bad epochs
    epochs = mne.Epochs(
        raw, events, event_id=mapping,
        tmin=0.0, tmax=4.0, baseline=None,
        preload=True, reject=dict(eeg=200e-6),
        verbose='WARNING'
    )
    
    # Fallback if too many epochs are rejected (need at least 15 for stable training/CV)
    if len(epochs) < 15:
        # Try a more relaxed threshold of 500uV
        epochs = mne.Epochs(
            raw, events, event_id=mapping,
            tmin=0.0, tmax=4.0, baseline=None,
            preload=True, reject=dict(eeg=500e-6),
            verbose='WARNING'
        )
        if len(epochs) < 15:
            # Fall back to no rejection if 500uV is still too strict
            epochs = mne.Epochs(
                raw, events, event_id=mapping,
                tmin=0.0, tmax=4.0, baseline=None,
                preload=True, reject=None,
                verbose='WARNING'
            )
    
    # 8. Extract raw epoch arrays and map labels to binary {T1: 1, T2: 2}
    X = epochs.get_data(copy=True)  # Shape: (n_epochs, n_channels, n_times)
    
    # Map to binary 1 and 2
    y = np.zeros(len(epochs), dtype=int)
    y[epochs.events[:, 2] == mapping['T1']] = 1
    y[epochs.events[:, 2] == mapping['T2']] = 2
    
    return X, y


class DWTFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor that computes Discrete Wavelet Transform (DWT) features.
    Can be used after CSP projection in the pipeline.
    """
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X: array-like of shape (n_epochs, n_channels, n_times)
        Returns: array-like of shape (n_epochs, n_channels * (level + 1))
        """
        n_epochs, n_channels, n_times = X.shape
        features_list = []
        
        for i in range(n_epochs):
            epoch_features = []
            for j in range(n_channels):
                channel_signal = X[i, j, :]
                # Compute DWT decomposition
                coeffs = pywt.wavedec(channel_signal, wavelet=self.wavelet, level=self.level)
                # Extract energy per level
                energy = [np.sum(c**2) for c in coeffs]
                # Log transform the energy to stabilize variance (similar to log-variance in CSP)
                log_energy = np.log(np.maximum(energy, 1e-10))
                epoch_features.extend(log_energy)
            features_list.append(epoch_features)
            
        return np.array(features_list)
