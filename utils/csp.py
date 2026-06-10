import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _compute_covariance(X_class):

    """
    Compute normalized spatial covariance for a class.
    
    Parameters:
    X_class (np.ndarray): Epochs for a specific class, shape (n_epochs, n_channels, n_times)
    
    Returns:
    sigma (np.ndarray): Mean normalized covariance, shape (n_channels, n_channels)
    """
    
    covs = []
    
    for epoch in X_class:
    
        cov = epoch @ epoch.T
        trace = np.trace(cov)
    
        if trace > 0:
            covs.append(cov / trace)
    
    return np.mean(covs, axis=0)


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
        sigma_1 = _compute_covariance(X[y == c1])
        sigma_2 = _compute_covariance(X[y == c2])

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

        # 6. Sort and select components by absolute distance from 0.5
        # The eigenvalues of S_1 range between 0 and 1.
        # Values close to 0 or 1 contain the most discriminative information.
        # Values close to 0.5 contain no discriminative information.
        sort_idx = np.argsort(np.abs(D - 0.5))[::-1]
        B_selected = B[:, sort_idx[:self.n_components]]

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
