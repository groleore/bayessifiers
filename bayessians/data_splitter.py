import numpy as np

from .distribution_estimator import DistributionEstimator


class ClassDataSubset():
    
    def __init__(self, data, target, label, **distribution_params):
        self.label = label
        self.values = data[target == label]
        self.all_values = data

        self.n_items = self.values.shape[0]
        self.n_features = self.values.shape[1]
        self.class_prob = self.n_items / data.shape[0]
        self.distribution_params = self._generate_distribution_params(**distribution_params)
    
    def get_calculator(self):
        if not hasattr(self, 'calculator'):
            self.calculator = DataCalculator(self.values)
        return self.calculator
    
    def get_distributions(self):
        if not hasattr(self, 'distributions'):
            data = self.values.to_numpy()
            data_features = np.array([C.flatten() for C in np.hsplit(data, self.n_features)])
            estimators = [DistributionEstimator(feature) for feature in data_features]
            self.distributions = [e.get_distribution(**p) for e, p in zip(estimators, self.distribution_params)]
        return self.distributions
    
    def _generate_distribution_params(self, **distribution_params):
        data = self.all_values.to_numpy()
        C = np.array([C.flatten() for C in np.hsplit(data, self.n_features)])
        return [ { **distribution_params, 'X_h': c } for c in C]


class DatasetSplitter():
    
    def __init__(self, X, y, **kwargs):
        self.data = X
        self.target = y
        
        self.labels = sorted(self.target.unique())
        self.n_items = self.data.shape[0]
        self.n_features = self.data.shape[1]
        
        self.kwargs = kwargs
        self.splitted_data = self.split_data()

    def split_data(self):
        return [ClassDataSubset(self.data, self.target, label, **self.kwargs) for label in self.labels]


class DataCalculator():
    
    def __init__(self, data):
        self.values = data
        self.n_items = self.values.shape[0]
        self.n_features = self.values.shape[1]
    
    def calculate_mean(self):
        return self.values.apply(np.mean, axis=0).values
    
    def calculate_features_number(self):
        return self.n_features
    
    def calculate_items_number(self):
        return self.n_items
    
    def calculate_centered_data(self):
        return self.values - self.calculate_mean()
      
    def calculate_var(self):
        return self.values.apply(np.var, axis=0).values
    
    def calculate_std(self):
        return self.values.apply(np.std, axis=0).values
    
    def calculate_covariance_matrix(self, denominator):
        data_centered = self.calculate_centered_data()
        U, s, Vh = np.linalg.svd(data_centered)
        s_2 = np.diag(s * s) / denominator
        
        padded_s_2 = np.zeros_like(Vh)
        padded_s_2[:s_2.shape[0],:s_2.shape[1]] = s_2
        
        return Vh.T @ padded_s_2 @ Vh
    
    def calculate_unnormalized_covariance_matrix(self):
        return self.calculate_covariance_matrix(self.n_items)
    
    def calculate_normalized_covariance_matrix(self):
        return self.calculate_covariance_matrix(self.n_items - 1)
    
    def calculate_correlation_matrix(self):
        covariance_matrix = self.calculate_normalized_covariance_matrix()
        diagonal = np.diag(1.0 / np.sqrt(covariance_matrix.diagonal()))
        return diagonal @ covariance_matrix @ diagonal
