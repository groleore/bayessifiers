import numpy as np
import math

from .abstract import AbstractBayesClassifier, AbstractClassPredictor


class DiscriminantAnalysisClassPredictor(AbstractClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)

        self.covariance_matrix = self.get_covariance_matrix(**kwargs)
        
        calculator = self.data_subset.get_calculator()
        self.mean = calculator.calculate_mean()
        self.n_features = calculator.calculate_features_number()
        self.covariance_inv = np.linalg.inv(self.covariance_matrix)

    def calculate_aposteriory_prob(self, X):
        pi_coef = pow(2 * math.pi, self.n_features)
        covariance_det = np.linalg.det(self.covariance_matrix)
        c = 1.0 / (pi_coef * math.sqrt(covariance_det))
        
        try:
            X_centered = X - self.mean
            exp_degree = X_centered.T @ self.covariance_inv @ X_centered
            exp = math.exp(-0.5 * exp_degree)
        except:
            exp = 1e-20
        
        return c * exp


class LDAClassifier(AbstractBayesClassifier):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)
    
    def init_class_predictors(self, **kwargs):
        pooled_covariance_matrix = self.calculate_pooled_covariance_matrix()
        splitted_data = self.data_splitter.splitted_data
        return [LDAClassPredictor(s_d, cov_m=pooled_covariance_matrix) for s_d in splitted_data]
    
    def get_class_predictor(self, data, **kwargs):
        return LDAClassPredictor(data, **kwargs)

    def calculate_pooled_covariance_matrix(self):
        n_features = self.data_splitter.n_features
        pooled_covariance_matrix = np.zeros((n_features, n_features))
        for d_s in self.data_splitter.splitted_data:
            covariance_matriix = d_s.get_calculator().calculate_unnormalized_covariance_matrix()
            pooled_covariance_matrix += covariance_matriix * d_s.class_prob
        return pooled_covariance_matrix
    
    def get_intercept(self):
        intercept = [predictor.calculate_intercept() for predictor in self.class_predictors]
        return np.stack(intercept, axis=0)
    
    def get_coef(self):
        coef = [predictor.calculate_coef() for predictor in self.class_predictors]
        return np.stack(coef, axis=0)
    
    @classmethod
    def get_name(cls):
        return 'LDA' 


class LDAClassPredictor(DiscriminantAnalysisClassPredictor):

    def get_covariance_matrix(self, **kwargs):
        return kwargs.get('cov_m')

    def calculate_coef(self):
        return np.linalg.inv(self.covariance_matrix) @ self.mean
    
    def calculate_intercept(self):
        apriori_log = np.log(self.calculate_apriori_prob())
        return -0.5 * self.mean.T @ np.linalg.inv(self.covariance_matrix) @ self.mean + apriori_log


class QDAClassifier(AbstractBayesClassifier):

    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)
    
    def get_class_predictor(self, data, **kwargs):
        return QDAClassPredictor(data, **kwargs)
    
    def get_covariances(self):
        splitted_data = self.data_splitter.splitted_data
        return [d_s.get_calculator().calculate_normalized_covariance_matrix() for d_s in splitted_data]
    
    @classmethod
    def get_name(cls):
        return 'QDA'


class QDAClassPredictor(DiscriminantAnalysisClassPredictor):

    def get_covariance_matrix(self, **kwargs):
        return self.data_subset.get_calculator().calculate_normalized_covariance_matrix()
