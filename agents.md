# AGENT: Total Perspective Vortex — EEG Motor Imagery BCI

## Mission

You are an autonomous coding agent implementing a complete Brain-Computer Interface (BCI) pipeline for the 42 school project *Total Perspective Vortex*. Your goal is a production-ready Python program (`mybci.py`) that classifies EEG motor imagery signals (hand vs. feet imagined movement) from the PhysioNet dataset using a custom dimensionality reduction algorithm integrated into a scikit-learn pipeline.

**Hard success criterion:** ≥60% mean classification accuracy across all 6 experiment types on held-out test data for all 109 subjects.

---

## Environment & Constraints

- **Language:** Python 3.10+
- **Core libraries:** `mne`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`
- **Bonus libraries:** `pywavelets` (PyWavelets) for wavelet preprocessing
- **Dataset:** PhysioNet EEG Motor Movement/Imagery (download via `mne.datasets.eegbci`)
  - 109 subjects, 64 EEG channels, 160 Hz sampling rate
  - Runs of interest: 3,4 (motor execution hand/feet), 5,6 (motor imagery hand/feet), 7,8, 9,10, 11,12, 13,14
  - Event markers: `T0` = rest, `T1` = left hand / both hands, `T2` = right hand / both feet
- **Do NOT** use `mne-realtime`
- **Do NOT** commit the dataset — only the Python source
- All sklearn custom components must subclass `BaseEstimator` and `TransformerMixin`

---

## Repository Structure to Produce

```
total-perspective-vortex/
├── mybci.py              # Main entry point (see CLI spec below)
├── preprocessing.py      # EEG loading, filtering, epoching, feature extraction
├── csp.py                # Custom CSP implementation (BaseEstimator + TransformerMixin)
├── pipeline.py           # Pipeline assembly, train/save/load logic
├── predict.py            # Stream-mode prediction logic
├── visualize.py          # Raw + filtered signal plots (call separately, not in train/predict)
├── utils.py              # Shared helpers (epoch splitting, scoring, file paths)
└── README.md             # Setup and usage instructions
```

---

## CLI Specification

```bash
# Train on subject S, run R — saves model to models/subj{S}_run{R}.pkl
python mybci.py <S> <R> train

# Predict on subject S, run R in streaming mode
python mybci.py <S> <R> predict

# Batch evaluation: all 109 subjects × 6 experiment types
python mybci.py
```

**train output format:**
```
[0.6666 0.4444 0.4444 ...] ← cross_val_score fold scores
cross_val_score: 0.XXXX
```

**predict output format (stream, ≤2s latency per epoch):**
```
epoch nb: [prediction] [truth] equal?
epoch 00: [2] [1] False
epoch 01: [1] [1] True
...
Accuracy: 0.XXXX
```

**batch output format:**
```
experiment 0: subject 001: accuracy = 0.X
...
Mean accuracy of the six different experiments for all 109 subjects:
experiment 0: accuracy = 0.XXXX
...
Mean accuracy of 6 experiments: 0.XXXX
```

---

## Implementation Tasks (execute in order)

### TASK 1 — Data Loading & Preprocessing (`preprocessing.py`)

1. Use `mne.datasets.eegbci.load_data(subject, runs)` to fetch `.edf` files
2. Concatenate raws: `mne.concatenate_raws(raws)`
3. Set standard montage: `mne.channels.make_standard_montage('standard_1005')`
4. **Bandpass filter** to mu + beta bands: 8–30 Hz (`raw.filter(8., 30.)`)
5. Extract events: `mne.events_from_annotations(raw)`
6. Epoch: `mne.Epochs(raw, events, event_id, tmin=0., tmax=4., baseline=None, preload=True)`
7. Drop bad epochs; map labels to binary `{T1: 1, T2: 2}`
8. **Feature extraction:** compute per-channel log-variance of epoched signal
   - Output shape: `(n_epochs, n_channels)` — this feeds directly into CSP
   - Alternative for bonus: Welch PSD per band per channel → `(n_epochs, n_channels * n_bands)`
9. Return `X` (epochs array, shape `n_epochs × n_channels × n_times`) and `y` (labels)

**Deliverable:** `load_and_preprocess(subject, runs) -> (X: np.ndarray, y: np.ndarray)`

---

### TASK 2 — Custom CSP Implementation (`csp.py`)

Implement Common Spatial Patterns from scratch. This is the mandatory core of the project.

```python
class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4): ...
    def fit(self, X, y): ...       # X: (n_epochs, n_channels, n_times)
    def transform(self, X): ...    # returns (n_epochs, n_components)
```

**Algorithm:**

1. For each class `c`, compute normalized spatial covariance:
   ```
   Σ_c = mean over epochs_c of (E @ E.T) / trace(E @ E.T)
   ```
   where `E` is the `(n_channels, n_times)` epoch matrix.

2. Composite covariance: `Σ_composite = Σ_1 + Σ_2`

3. Whitening: eigen-decompose `Σ_composite = V Λ V^T`, compute whitening matrix `P = Λ^{-1/2} V^T`

4. Transform class covariances: `S_c = P Σ_c P^T`

5. Generalized eigen-decomp of `S_1`:
   ```
   S_1 B = B D    (scipy.linalg.eigh or np.linalg.eigh)
   ```
   Eigenvectors sorted by eigenvalue (largest first for class 1, smallest = largest for class 2).

6. Projection matrix: `W = B^T P`  (shape: `n_components × n_channels`)

7. `transform`: apply `W` to each epoch → extract `n_components` spatial filters, compute log-variance of each filtered signal → feature vector per epoch.

**Allowed numpy/scipy:** `np.linalg.eigh`, `scipy.linalg.eigh`, `np.cov`, `np.linalg.svd`

**Forbidden:** `sklearn.decomposition.CSP` (must be your own implementation for mandatory part)

---

### TASK 3 — Pipeline Assembly (`pipeline.py`)

```python
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle

def build_pipeline():
    return Pipeline([
        ('csp', CSP(n_components=4)),
        ('clf', LinearDiscriminantAnalysis()),   # LDA is standard for CSP+BCI
    ])

def train(subject, runs):
    X, y = load_and_preprocess(subject, runs)
    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=10, shuffle=True)   # shuffle=True, no fixed seed
    scores = cross_val_score(pipe, X, y, cv=cv)
    print(scores)
    print(f"cross_val_score: {scores.mean():.4f}")
    pipe.fit(X, y)
    os.makedirs('models', exist_ok=True)
    with open(f'models/subj{subject:03d}_run{runs[0]:02d}.pkl', 'wb') as f:
        pickle.dump(pipe, f)
```

**Split strategy:**
- Use `StratifiedKFold` with `shuffle=True` and no fixed random seed (different splits each run → no overfitting to split)
- Hold out final 20% as test set before any CV (use `train_test_split` stratified)
- Report `cross_val_score` on training portion only; final accuracy on test set

---

### TASK 4 — Streaming Prediction (`predict.py`)

```python
def predict_stream(subject, runs):
    pipe = load_model(subject, runs)
    X, y = load_and_preprocess(subject, runs)
    # Simulate stream: iterate epoch by epoch with ≤2s wall-clock latency
    correct = 0
    for i, (epoch, label) in enumerate(zip(X, y)):
        t0 = time.time()
        pred = pipe.predict(epoch[np.newaxis, ...])[0]
        elapsed = time.time() - t0
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)   # pad to simulate real-time gap
        match = pred == label
        correct += int(match)
        print(f"epoch {i:02d}: [{pred}] [{label}] {'True' if match else 'False'}")
    print(f"Accuracy: {correct / len(y):.4f}")
```

---

### TASK 5 — Batch Evaluation (`mybci.py` no-args mode)

Define the 6 experiment types as run pairs:
```python
EXPERIMENTS = {
    0: [3, 4],    # motor execution: hand vs feet
    1: [5, 6],    # motor imagery: hand vs feet
    2: [7, 8],
    3: [9, 10],
    4: [11, 12],
    5: [13, 14],
}
```

For each experiment × each subject (1–109):
- `load_and_preprocess(subject, runs)`
- `build_pipeline().fit(X_train, y_train)`
- score on `X_test`
- Aggregate per experiment, print mean

---

### TASK 6 — Visualization (`visualize.py`)

Produce and save (do not block execution):
1. Raw signal plot (first 10s, 5 channels) before filtering
2. Filtered signal plot (same window, same channels) after bandpass
3. Power spectral density plot (pre vs post filter overlay)
4. Topographic map of CSP filters (use `mne.viz.plot_topomap`)

Call with: `python visualize.py <subject> <run>`

---

## Bonus Tasks (implement after mandatory ≥60% is verified)

### BONUS 1 — Wavelet Preprocessing (`preprocessing.py`)

Replace or augment bandpass + log-variance features with discrete wavelet transform features:

```python
import pywt
# Per epoch, per channel: DWT decomposition
coeffs = pywt.wavedec(channel_signal, wavelet='db4', level=4)
# Extract energy per level (approximation + details)
features = [np.sum(c**2) for c in coeffs]
```

Combine wavelet energy features with CSP spatial filtering. Expected accuracy improvement: +2–5%.

### BONUS 2 — Custom LDA Classifier (`lda.py`)

Implement `LinearDiscriminantAnalysis` from scratch as `BaseEstimator + ClassifierMixin`:
- Compute class means, shared covariance `Σ_w` (within-class)
- Decision boundary: `w = Σ_w^{-1} (μ_1 - μ_2)`, threshold at midpoint
- Use `scipy.linalg.solve` instead of explicit inverse

### BONUS 3 — Custom Eigenvalue Solver (hard)

Implement the QR algorithm for symmetric eigendecomposition without `np.linalg.eigh`:
- QR iteration with Householder reflections for tridiagonalization
- Implicit shift for convergence acceleration
- Validate against `np.linalg.eigh` to tolerance `1e-6`

This replaces the numpy/scipy calls inside `csp.py`.

### BONUS 4 — Riemannian Geometry Classifier

Replace CSP+LDA with minimum distance to mean (MDM) in the Riemannian manifold of SPD matrices:
- Compute Fréchet mean of covariance matrices per class
- Classify by geodesic distance to class means
- Reference: `pyriemann` library (study the math, implement from scratch)

### BONUS 5 — Additional Dataset

Integrate BCI Competition IV Dataset 2a (4-class MI, 22 channels):
- Adapt `load_and_preprocess` to handle `.gdf` format via MNE
- Extend pipeline to 4-class (OvR strategy with 4 binary CSPs)
- Report accuracy on this dataset separately

---

## Accuracy Targets & Validation Protocol

| Checkpoint | Target |
|---|---|
| Single subject, best run | ≥70% on test split |
| All subjects, experiment 0 (execution) | ≥60% mean |
| All subjects, all 6 experiments | ≥60% mean |
| With wavelet bonus | ≥65% mean |

**Anti-overfitting rules:**
- `shuffle=True` in CV, no `random_state` fixed globally
- Train/test split before any CV computation
- Never tune hyperparameters on the test set

---

## Agent Execution Protocol

1. **Verify environment:** `pip install mne scikit-learn scipy numpy matplotlib pywavelets`
2. **Download dataset:** run `mne.datasets.eegbci.load_data(1, [3,4])` to confirm connectivity
3. **Implement in order:** TASK 1 → TASK 2 → TASK 3 → TASK 4 → TASK 5 → TASK 6
4. **Test gate after each task:** run the relevant script, confirm output shape/format before proceeding
5. **Accuracy gate:** after TASK 5, run full batch eval. If mean accuracy <60%, debug in this order:
   - Check epoch label mapping (T1/T2 → 1/2)
   - Increase `n_components` in CSP (try 4, 6, 8)
   - Switch classifier: try `SVC(kernel='rbf', C=1.0)` or `RandomForestClassifier`
   - Extend tmin/tmax epoch window (try tmin=0.5, tmax=3.5 to avoid onset artifacts)
6. **After mandatory passing:** implement BONUS 1 (wavelets) first — highest ROI
7. **Commit only source**, never `data/` or `models/`

---

## Key Implementation Notes

- CSP requires `X` in shape `(n_epochs, n_channels, n_times)` — verify before fitting
- MNE epochs `.get_data()` returns this shape by default
- For binary classification, CSP with `n_components=4` uses 2 filters per class (first 2 + last 2 eigenvalues) — implement this selection explicitly
- LDA is strongly preferred over SVM for CSP output — the features are already maximally discriminative; SVM adds marginal benefit but slower
- Epoch rejection: use `reject=dict(eeg=200e-6)` in `mne.Epochs` to auto-drop high-amplitude artifacts
- Normalize covariance matrices by trace before accumulation (crucial for CSP stability)