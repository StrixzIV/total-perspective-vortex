import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomLDA(BaseEstimator, ClassifierMixin):
    """
    Custom Linear Discriminant Analysis (LDA) classifier implemented from scratch.
    Fits scikit-learn API: inherits from BaseEstimator and ClassifierMixin.
    """
    def __init__(self, regularization=1e-6):
        self.regularization = regularization
        self.classes_ = None
        self.coef_ = None  # Weight vector w
        self.intercept_ = None  # Bias b
        self.mu_1_ = None
        self.mu_2_ = None

    def fit(self, X, y):
        """
        Fit the CustomLDA model.
        
        Parameters:
        X (np.ndarray): Shape (n_samples, n_features)
        y (np.ndarray): Labels, shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"CustomLDA only supports binary classification. Found classes: {self.classes_}")

        c1, c2 = self.classes_[0], self.classes_[1]
        
        X1 = X[y == c1]
        X2 = X[y == c2]

        N1 = len(X1)
        N2 = len(X2)
        N = len(X)

        if N1 == 0 or N2 == 0:
            raise ValueError("Each class must have at least one sample.")

        # 1. Compute class means
        self.mu_1_ = np.mean(X1, axis=0)
        self.mu_2_ = np.mean(X2, axis=0)

        # 2. Compute within-class covariance
        # (X1 - mu1).T @ (X1 - mu1) is N1 * cov1
        F = X.shape[1]
        cov1 = np.cov(X1, rowvar=False, ddof=0) if N1 > 1 else np.zeros((F, F))
        cov2 = np.cov(X2, rowvar=False, ddof=0) if N2 > 1 else np.zeros((F, F))

        # Handle 1D case (single feature)
        if F == 1:
            cov1 = np.atleast_2d(cov1)
            cov2 = np.atleast_2d(cov2)

        # Pool covariance
        Sigma_w = (N1 * cov1 + N2 * cov2) / (N - 2)
        
        # Add ridge regularization
        Sigma_w += self.regularization * np.eye(F)

        # 3. Solve for weight vector w: Sigma_w @ w = mu_1 - mu_2
        # Use scipy.linalg.solve for numerical stability
        self.coef_ = scipy.linalg.solve(Sigma_w, self.mu_1_ - self.mu_2_)

        # 4. Compute bias term b
        self.intercept_ = -0.5 * np.dot(self.coef_, self.mu_1_ + self.mu_2_)

        return self

    def decision_function(self, X):
        """
        Compute decision values.
        
        Parameters:
        X (np.ndarray): Shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Decision values of shape (n_samples,)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.coef_) + self.intercept_

    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        X (np.ndarray): Shape (n_samples, n_features)
        
        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,)
        """
        decisions = self.decision_function(X)
        preds = np.where(decisions >= 0.0, self.classes_[0], self.classes_[1])
        return preds
