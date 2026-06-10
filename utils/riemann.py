import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def _matrix_function(A, fn):
    """
    Apply a scalar function to the eigenvalues of a symmetric matrix A.
    A = V * fn(Lambda) * V.T
    """
    eigenvals, V = np.linalg.eigh(A)
    # fn is applied to the eigenvalues, reconstructed back to matrix space.
    return (V * fn(eigenvals)) @ V.T

def matrix_log(A):
    """Matrix logarithm of symmetric positive definite matrix A."""
    return _matrix_function(A, lambda x: np.log(np.maximum(x, 1e-12)))

def matrix_exp(A):
    """Matrix exponential of symmetric matrix A."""
    return _matrix_function(A, np.exp)

def matrix_sqrt(A):
    """Matrix square root of symmetric positive semi-definite matrix A."""
    return _matrix_function(A, lambda x: np.sqrt(np.maximum(x, 0.0)))

def matrix_invsqrt(A):
    """Inverse matrix square root of symmetric positive definite matrix A."""
    return _matrix_function(A, lambda x: 1.0 / np.sqrt(np.maximum(x, 1e-12)))

def geodesic_distance(A, B):
    """
    Calculate the Riemannian geodesic distance between two SPD matrices A and B.
    d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F
    """
    A_invsqrt = matrix_invsqrt(A)
    tmp = A_invsqrt @ B @ A_invsqrt
    log_tmp = matrix_log(tmp)
    return np.linalg.norm(log_tmp, ord='fro')

def riemannian_mean(covariances, tol=1e-5, max_iter=10):
    """
    Compute the Fréchet mean of a set of SPD matrices iteratively.
    
    Parameters:
    covariances (np.ndarray): Covariances array of shape (n_matrices, n_channels, n_channels)
    tol (float): Convergence tolerance.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    np.ndarray: Fréchet mean covariance of shape (n_channels, n_channels)
    """
    N, C, _ = covariances.shape
    if N == 0:
        raise ValueError("Must provide at least one covariance matrix.")
    if N == 1:
        return covariances[0]

    # Initialize mean M with the arithmetic mean of covariances
    M = np.mean(covariances, axis=0)
    # Force symmetry
    M = 0.5 * (M + M.T)

    for _ in range(max_iter):
        M_sqrt = matrix_sqrt(M)
        M_invsqrt = matrix_invsqrt(M)

        # Compute average of mapped matrices log(M^-1/2 * P_i * M^-1/2)
        sum_log = np.zeros((C, C))
        for P in covariances:
            tmp = M_invsqrt @ P @ M_invsqrt
            sum_log += matrix_log(tmp)
        
        J = sum_log / N
        norm = np.linalg.norm(J, ord='fro')

        # Check convergence
        if norm < tol:
            break

        # M_next = M_sqrt @ exp(J) @ M_sqrt
        M = M_sqrt @ matrix_exp(J) @ M_sqrt
        # Force symmetry
        M = 0.5 * (M + M.T)

    return M

def compute_epochs_covariances(X, shrink=0.2):
    """
    Compute trace-normalized spatial covariances for each epoch, applying shrinkage regularization.
    X: shape (n_epochs, n_channels, n_times)
    Returns: covariances array of shape (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, n_times = X.shape
    covs = np.zeros((n_epochs, n_channels, n_channels))
    for i in range(n_epochs):
        E = X[i]
        cov = E @ E.T
        trace = np.trace(cov)
        if trace > 0:
            P = cov / trace
        else:
            P = np.eye(n_channels) / n_channels
            
        if shrink > 0:
            P = (1 - shrink) * P + shrink * (np.eye(n_channels) / n_channels)
        covs[i] = P
    return covs


class RiemannianMDM(BaseEstimator, ClassifierMixin):
    """
    Minimum Distance to Mean (MDM) classifier in the Riemannian manifold of SPD matrices.
    Bypasses CSP by computing spatial covariance matrices of epochs and classifying based on
    geodesic distance to the class Riemannian Fréchet means.
    """
    def __init__(self, shrink=0.2, tol=1e-5, max_iter=10):
        self.shrink = shrink
        self.tol = tol
        self.max_iter = max_iter
        self.classes_ = None
        self.means_ = {}

    def fit(self, X, y):
        """
        Fit the Riemannian MDM model.
        
        Parameters:
        X (np.ndarray): Epochs array of shape (n_epochs, n_channels, n_times)
        y (np.ndarray): Labels, shape (n_epochs,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"RiemannianMDM only supports binary classification. Found classes: {self.classes_}")

        # Compute covariances for all epochs
        covs = compute_epochs_covariances(X, shrink=self.shrink)

        # Compute Riemannian mean for each class
        for c in self.classes_:
            class_covs = covs[y == c]
            if len(class_covs) == 0:
                raise ValueError(f"Class {c} has no samples.")
            self.means_[c] = riemannian_mean(class_covs, tol=self.tol, max_iter=self.max_iter)

        return self

    def predict(self, X):
        """
        Predict labels for the given epochs.
        
        Parameters:
        X (np.ndarray): Epochs array of shape (n_epochs, n_channels, n_times)
        
        Returns:
        np.ndarray: Predicted labels of shape (n_epochs,)
        """
        X = np.asarray(X, dtype=float)
        covs = compute_epochs_covariances(X, shrink=self.shrink)
        
        preds = []
        for cov in covs:
            # Compute geodesic distance to the mean of each class
            dist_1 = geodesic_distance(cov, self.means_[self.classes_[0]])
            dist_2 = geodesic_distance(cov, self.means_[self.classes_[1]])
            
            # Predict the class with the minimum distance
            if dist_1 <= dist_2:
                preds.append(self.classes_[0])
            else:
                preds.append(self.classes_[1])
                
        return np.array(preds)
