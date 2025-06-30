import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

class InformationGeometryTransformer(BaseEstimator, TransformerMixin):
    """
    Performs a supervised, information geometry-inspired dimensionality reduction.

    This transformer finds a set of orthogonal basis vectors in the feature space
    that are iteratively chosen to be maximally informative about a target variable 'y'.
    The process is akin to a form of forward feature selection in a projected space.

    Parameters
    ----------
    n_components : int, default=2
        The number of components (dimensions) to keep.

    Attributes
    ----------
    components_ : ndarray of shape (n_features, n_components)
        The principal axes in feature space, representing the directions of
        maximum information about the target.

    explained_variance_ratio_ : ndarray of shape (n_components,)
        The amount of variance in the target 'y' explained by each of the selected
        components.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit the model with X and y.

        Args:
            X (np.ndarray): Input data of shape [n_samples, n_features].
            y (np.ndarray): Target data of shape [n_samples,].
        
        Returns:
            self: object
                Returns the instance itself.
        """
        X, y = check_X_y(X, y, y_numeric=True)
        y = y.reshape(-1, 1)

        # It's crucial for this method to work with standardized data
        self.scaler_X_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y)
        X_scaled = self.scaler_X_.transform(X)
        y_scaled = self.scaler_y_.transform(y)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        basis_vectors = []
        residual = y_scaled.copy()
        
        total_y_variance = np.var(y_scaled)
        explained_variance_ratios = []

        for i in range(self.n_components):
            correlations = np.array([
                np.corrcoef(X_scaled[:, j], residual.ravel())[0, 1] if np.var(X_scaled[:, j]) > 1e-9 else 0
                for j in range(n_features)
            ])
            correlations = np.nan_to_num(correlations)
            
            selected_feature_idx = np.argmax(np.abs(correlations))
            X_selected = X_scaled[:, selected_feature_idx].reshape(-1, 1)

            beta = np.linalg.pinv(X_selected.T @ X_selected) @ X_selected.T @ residual
            predictions = X_selected @ beta

            grad_direction = X_scaled.T @ (residual - predictions)

            for v_prev in basis_vectors:
                proj = (grad_direction.T @ v_prev / (v_prev.T @ v_prev)) * v_prev
                grad_direction -= proj

            norm = np.linalg.norm(grad_direction)
            if norm > 1e-9:
                grad_direction /= norm
            else:
                grad_direction = np.zeros_like(grad_direction)

            basis_vectors.append(grad_direction.reshape(-1, 1))

            new_component = X_scaled @ grad_direction
            
            # Variance of the residual explained by this new component
            explained_variance = np.var(y_scaled) - np.var(residual)
            explained_variance_ratios.append(explained_variance / total_y_variance)

            # Update residual
            beta_residual = (np.linalg.pinv(new_component.T @ new_component) @ new_component.T @ residual)
            residual -= new_component * beta_residual

        self.components_ = np.hstack(basis_vectors)
        self.explained_variance_ratio_ = np.array(explained_variance_ratios)

        return self

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        Args:
            X (np.ndarray): Data to transform of shape [n_samples, n_features].

        Returns:
            np.ndarray: Transformed data of shape [n_samples, n_components].
        """
        check_is_fitted(self)
        X = check_array(X)
        X_scaled = self.scaler_X_.transform(X)
        return X_scaled @ self.components_

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Args:
            X_transformed (np.ndarray): Data to inverse transform of shape [n_samples, n_components].

        Returns:
            np.ndarray: Reconstructed data of shape [n_samples, n_features].
        """
        check_is_fitted(self)
        X_reconstructed_scaled = X_transformed @ self.components_.T
        return self.scaler_X_.inverse_transform(X_reconstructed_scaled)

