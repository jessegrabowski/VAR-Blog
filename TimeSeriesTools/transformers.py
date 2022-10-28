from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X
    def inverse_transform(self, X, y=None):
        return X
        
        
class DifferenceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self._x0 = None
        self.columns = columns

    def fit(self, X, y=None):
        self._validate_columns(X)
        _X = X[self.columns].copy()
        self.x0 = _X[_X.notna() & _X.diff().isna()].copy()
        return self

    def transform(self, X, y=None):
        new_X = X.copy()
        new_X[self.columns] = X[self.columns].diff()
        return new_X
    
    def inverse_transform(self, X, y=None):
        new_X = X.copy()
        new_X[self.columns] = X[self.columns].fillna(self.x0).cumsum()
        return new_X
        
    def _validate_columns(self, X):
        if self.columns is None:
            self.columns = X.columns
        else:
            assert all([col in X.columns for col in self.columns])
    
class DetrendTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, trend = 'c', columns=None):
        self.params = None
        self.trend = trend
        self.columns = columns

    def _build_feature_matrix(self, X):
        trend = self.trend
        T = X.shape[0]
        features = None
        
        if 'c' in trend:
            features = np.ones(T)[:, None]
        if 't' in trend:
            lin_trend = np.arange(T)[:, None]
            features = lin_trend if features is None else np.c_[features, lin_trend]
        if 'tt' in trend:
            quad_trend = np.arange(T)[:, None] ** 2
            features = quad_trend if features is None else np.c_[features, quad_trend]
        
        return features
        
    def fit(self, X, y=None):
        features = self._build_feature_matrix(X)
        self._validate_columns(X)
        
        def regress(endog, exog):
            nan_mask = ~np.isnan(endog)
            return np.linalg.solve(exog[nan_mask].T @ exog[nan_mask], exog[nan_mask].T @ endog[nan_mask]) 
        
        params = np.apply_along_axis(regress, axis=0, arr=X[self.columns].values, exog=features)
        self.params = np.atleast_1d(np.c_[params])
        
        return self
    
    def transform(self, X, y=None):
        n = 1 if len(self.columns) == 1 else len(self.columns)
        features = self._build_feature_matrix(X)
        
        new_X = X.copy()
        X_hat = np.einsum('tkn, kn->tn', np.dstack([features] *  n), self.params)
        new_X[self.columns] = new_X[self.columns] - X_hat
        return new_X
    
    def inverse_transform(self, X, y=None):
        # X are residuals (output of transform)
        n = 1 if len(self.columns) == 1 else len(self.columns)
        features = self._build_feature_matrix(X)
        new_X = X.copy()

        X_hat = np.einsum('tkn, kn->tn', np.dstack([features] *  n), self.params)
        
        new_X[self.columns] = new_X[self.columns] + X_hat

        return new_X

    def _validate_columns(self, X):
        if self.columns is None:
            self.columns = X.columns
        else:
            assert all([col in X.columns for col in self.columns])

        
class LogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, lamb=0, columns=None):
        self.lamb = lamb
        self.signs = None
        self.columns = columns

    def fit(self, X, y=None):
        self._validate_columns(X)
        return self
    
    def transform(self, X, y=None):
        lamb = self.lamb
        new_X = X.copy()
        if lamb == 0:
            new_X[self.columns] = X[self.columns].apply(np.log)
            return new_X
        
        new_X[self.columns] = X[self.columns].apply(lambda x: (np.sign(x) * np.abs(x) ** lamb - 1)  / lamb)
        return new_X
    
    def inverse_transform(self, X, y=None):
        lamb = self.lamb    
        new_X = X.copy()

        if lamb == 0:
            new_X[self.columns] = X[self.columns].apply(np.exp)
            return new_X
        
        new_X[self.columns] = X[self.columns].apply(lambda x: np.sign(lamb * x + 1) * np.abs(lamb * x + 1) ** (1 / lamb))
        return new_X

    def _validate_columns(self, X):
        if self.columns is None:
            self.columns = X.columns
        else:
            assert all([col in X.columns for col in self.columns])

class PandasStandardScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None):
        self.means = None
        self.stds = None
        self.columns = columns
        
    def fit(self, X, y=None):
        self._validate_columns(X)
        self.means = X[self.columns].mean()
        self.stds = X[self.columns].std()
        
        return self
    
    def transform(self, X, y=None):
        means = self.means
        stds = self.stds
        
        new_X = X.copy()
        new_X[self.columns] = (X[self.columns] - means) / stds
        
        return new_X
    
    def inverse_transform(self, X, y=None):
        means = self.means
        stds = self.stds
        new_X = X.copy()
        
        new_X[self.columns] = stds * X[self.columns] + means

        return new_X

    def _validate_columns(self, X):
        if self.columns is None:
            self.columns = X.columns
        else:
            assert all([col in X.columns for col in self.columns])

    
class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None):
        self.mins = None
        self.maxes = None
        self.columns = columns
        
    def fit(self, X, y=None):
        self._validate_columns(X)
        self.mins = X[self.columns].min()
        self.maxes = X[self.columns].max()
        
        return self
    
    def transform(self, X, y=None):
        mins = self.mins
        maxes = self.maxes
        
        new_X = X.copy()
        new_X[self.columns] = (X[self.columns] - mins) / (maxes - mins)
        
        return new_X
    
    def inverse_transform(self, X, y=None):
        mins  = self.mins
        maxes = self.maxes
        
        new_X = X.copy()
        new_X[self.columns] = (maxes - mins) * X[self.columns] + maxes

        return new_X

    def _validate_columns(self, X):
        if self.columns is None:
            self.columns = X.columns
        else:
            assert all([col in X.columns for col in self.columns])