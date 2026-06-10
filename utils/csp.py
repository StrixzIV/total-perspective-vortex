import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):

    """
    Common Spatial Patterns (CSP) for binary classification of EEG signals.
    Subclasses BaseEstimator and TransformerMixin for seamless integration in scikit-learn pipelines.
    """

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None  # Projection matrix W

    def fit(self, X, y):

        """
        Fit the CSP projection matrix.
        
        Parameters:
        X (np.ndarray): Shape (n_epochs, n_channels, n_times)
        y (np.ndarray): Labels (binary class labels)
        """

        # Validate inputs
        n_epochs, n_channels, n_times = X.shape
        classes = np.unique(y)

        if len(classes) != 2:
            raise ValueError(f"CSP only supports binary classification. Found classes: {classes}")

        c1, c2 = classes[0], classes[1]

        # 1. Compute normalized spatial covariance per class
        covs_1 = []
        for epoch in X[y == c1]:
            cov = epoch @ epoch.T
            trace = np.trace(cov)
            if trace > 0:
                covs_1.append(cov / trace)
        sigma_1 = np.mean(covs_1, axis=0)

        covs_2 = []
        for epoch in X[y == c2]:
            cov = epoch @ epoch.T
            trace = np.trace(cov)
            if trace > 0:
                covs_2.append(cov / trace)
        sigma_2 = np.mean(covs_2, axis=0)

        # 2. Composite covariance
        sigma_composite = sigma_1 + sigma_2

        # 3. Whitening
        eigenvals, V = np.linalg.eigh(sigma_composite)
        # Avoid division by zero/negative values
        eigenvals = np.maximum(eigenvals, 1e-10)
        # P = diag(Λ^{-1/2}) @ V.T
        P = np.diag(1.0 / np.sqrt(eigenvals)) @ V.T

        # 4. Transform class covariances
        S_1 = P @ sigma_1 @ P.T

        # 5. Generalized eigen-decomposition of S_1
        # S_1 @ B = B @ D
        D, B = np.linalg.eigh(S_1)

        # 6. Sort and select components
        # np.linalg.eigh returns eigenvalues in ascending order (0 to 1).
        # We need the first k (smallest eigenvalues of S_1 = largest of S_2)
        # and the last k (largest eigenvalues of S_1 = smallest of S_2)
        k = self.n_components // 2

        # Largest S1 eigenvalues first (descending: D[-1], D[-2], ..., D[-k])
        part_c1 = B[:, -1:-k-1:-1]
        # Smallest S1 eigenvalues first (ascending: D[0], D[1], ..., D[k-1])
        part_c2 = B[:, :k]

        B_selected = np.concatenate([part_c1, part_c2], axis=1)

        # 7. Projection matrix W = B_selected.T @ P
        self.filters_ = B_selected.T @ P

        return self

    def transform(self, X):

        """
        Apply CSP filters and extract features.
        
        Parameters:
        X (np.ndarray): Shape (n_epochs, n_channels, n_times)
        
        Returns:
        X_features (np.ndarray): Shape (n_epochs, n_components) (log-variance of components)
        """

        if self.filters_ is None:
            raise RuntimeError("CSP is not fitted yet. Call fit first.")
            
        # Z = W @ E -> shape: (n_epochs, n_components, n_times)
        Z = np.matmul(self.filters_, X)

        # Compute variance along the time dimension
        var = np.var(Z, axis=-1)

        # Return log-variance
        return np.log(np.maximum(var, 1e-10))
