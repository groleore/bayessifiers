import numpy as np
from .abstract import AbstractBayesClassifier, AbstractCopulaClassPredictor

class ArchimedianCopulaClassifier(AbstractBayesClassifier):
    pass

class AbstractArchimedianCopulaClassPredictor(AbstractCopulaClassPredictor):

    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)
        
        self.alpha = kwargs.get('alpha', 0.5)
        self.eps = 1e-20
        
        calculator = self.data_subset.get_calculator()
        self.n_features = calculator.calculate_features_number()


class ClaytonCopulaClassifier(ArchimedianCopulaClassifier):
            
    def get_class_predictor(self, labeled_data, **kwargs):
        return ClaytonCopulaClassPredictor(labeled_data, **kwargs)

    @classmethod
    def get_name(cls):
        return 'Clayton copula' 


class ClaytonCopulaClassPredictor(AbstractArchimedianCopulaClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)
    
    def calculate_copula_density(self, X):
        alpha = (-1.0) * self.alpha
        U = self.calculate_features_distribution(X)
        
        m_U = np.multiply.reduce(U)
        m_U = pow(max(m_U, self.eps), alpha - 1.0)
        
        s_U = [pow(max(u, self.eps), alpha) for u in U]
        s_U = np.sum(s_U) - self.n_features + 1.0
        s_U = pow(s_U, 1.0 / alpha - self.n_features)

        return m_U * s_U


class FrankCopulaClassifier(ArchimedianCopulaClassifier):
    
    def get_class_predictor(self, labeled_data, **kwargs):
        return FrankCopulaClassPredictor(labeled_data, **kwargs)

    @classmethod
    def get_name(cls):
        return 'Frank copula'


class FrankCopulaClassPredictor(AbstractArchimedianCopulaClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)

        if self.n_features != 2:
            raise Exception('Frank copula is only for 2-dimensional data')
            
    def calculate_copula_density(self, X):
        alpha = (-1.0) * self.alpha
        
        U = self.calculate_features_distribution(X)
        u_1, u_2 = U
        
        n = alpha * (np.exp(alpha) - 1.0) * np.exp(alpha * (u_1 + u_2))
        
        d1 = np.exp(alpha * u_1) - 1.0
        d2 = np.exp(alpha * u_2) - 1.0
        d = d1 * d2 + np.exp(alpha) - 1.0
        d = pow(d, 2.0)
 
        return n / d


class GumbelCopulaClassifier(ArchimedianCopulaClassifier):

    def get_class_predictor(self, labeled_data, **kwargs):
        return GumbelCopulaClassPredictor(labeled_data, **kwargs)

    @classmethod
    def get_name(cls):
        return 'Gumbel Copula'


class GumbelCopulaClassPredictor(AbstractArchimedianCopulaClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)

        if self.alpha < 1.0:
            raise Exception('alpha must be non-less than one')

        if self.n_features != 2:
            raise Exception('Gumbel copula is only for 2-dimensional data')
            
    def calculate_copula_density(self, X):
        alpha = self.alpha
        
        U = self.calculate_features_distribution(X)
        u_1, u_2 = U
        
        if abs(u_1) < self.eps or abs(u_2) < self.eps:
            return 0.0
        
        if u_1 == 1.0 and u_2 == 1.0:
            return 0.0
        
        log_u_1 = pow((-1) * np.log(u_1), alpha)
        log_u_2 = pow((-1) * np.log(u_2), alpha)
        sum_log = log_u_1 + log_u_2
        C = np.exp((-1) * pow(sum_log, 1.0 / alpha))
        
        inv = 1.0 / (u_1 * u_2)
        sum_log_p_2 = pow(sum_log, 1.0 / alpha - 2.0)
        log_m = pow(np.log(u_1) * np.log(u_2), alpha - 1)
        
        sum_log_p_minus = pow(sum_log, -1.0 / alpha) + alpha - 1.0
        return C * inv * sum_log_p_2 * log_m * sum_log_p_minus
