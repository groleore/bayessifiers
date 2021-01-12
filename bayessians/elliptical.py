import numpy as np
from scipy import stats
from scipy.special import gamma

from .abstract import AbstractBayesClassifier, AbstractCopulaClassPredictor

class EllipticalCopulaClassifier(AbstractBayesClassifier):
    
    def get_correlation_matrixes(self):
        correlation_matrix = [predictor.correlation_matrix for predictor in self.class_predictors]
        return np.stack(correlation_matrix, axis=0)


class EllipticalCopulaClassPredictor(AbstractCopulaClassPredictor):

    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)

        calculator = self.data_subset.get_calculator()
        self.n_features = calculator.calculate_features_number()
        self.n_items = calculator.calculate_items_number()

        self.correlation_matrix = self.calculate_correlation_matrix()
        self.correlation_det = np.linalg.det(self.correlation_matrix)
        self.correlation_inv = np.linalg.inv(self.correlation_matrix)
        with np.errstate(invalid='raise'):
            try:
                self.c = 1.0 / np.sqrt(self.correlation_det)
            except:
                print('Cannot calculate c')
                self.c = 0.0


    def calculate_correlation_matrix(self):
        cov_matrix = np.zeros((self.n_features, self.n_features), dtype=float)
        for row in self.data_subset.values.values:
            sigma = self.calculate_features_distribution(row)
            sigma = stats.norm.ppf(sigma).reshape(1, -1)
            cov_matrix += sigma.T @ sigma
        cov_matrix = (1.0 / self.n_items) * cov_matrix
        diagonal = np.diag(1.0 / np.sqrt(cov_matrix.diagonal()))
        return diagonal @ cov_matrix @ diagonal


class GaussianCopulaClassifier(EllipticalCopulaClassifier):

    def get_class_predictor(self, labeled_data, **kwargs):
        return GaussianCopulaClassPredictor(labeled_data, **kwargs)
        
    @classmethod
    def get_name(cls):
        return 'Gaussian copula'  


class GaussianCopulaClassPredictor(EllipticalCopulaClassPredictor):

    def calculate_copula_density(self, X):
        U = self.calculate_features_distribution(X)
        U = stats.norm.ppf(U).reshape(1, -1)

        identity_matrix = np.identity(self.n_features)
        exp_degree = U @ (self.correlation_inv - identity_matrix) @ U.T
        exp_degree = exp_degree[0][0]
        with np.errstate(invalid='raise'):
            try:
                exp = np.exp(-0.5 * exp_degree)
            except:
                exp = 0.0
        
        return self.c * exp


class StudentCopulaClassifier(EllipticalCopulaClassifier):

    def get_class_predictor(self, labeled_data, **kwargs):
        return StudentCopulaClassPredictor(labeled_data, **kwargs)

    @classmethod
    def get_name(cls):
        return 'Student Copula'


class StudentCopulaClassPredictor(EllipticalCopulaClassPredictor):

    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)

        v = self.calculate_v()
        self.init_v(v)
        self.correlation_matrix = self.calculate_t_correlation_matrix()

    def init_v(self, v):
        self.v = v

        self.half_v_n = (self.v + self.n_features) / 2.0
        self.half_items = (self.v + 1) / 2.0
        
        g_1 = gamma(self.half_v_n)
        g_2 = gamma(self.v / 2.0)
        g_3 = gamma(self.half_items)
        
        with np.errstate(invalid='raise'):
            try:
                self.g = g_1 / g_2
                self.g_n = pow(g_2 / g_3, self.n_features)
            except:
                self.g  = 1.0
                self.g_n = 1.0

    def calculate_copula_density(self, X):
        U = self.calculate_features_distribution(X)
        U = stats.t.ppf(U, self.v).reshape(1, -1)
        
        with np.errstate(invalid='raise'):
            U_corr = U @ self.correlation_inv @ U.T
            U_corr = U_corr[0][0]
            U_corr = 1.0 + U_corr / self.v
            U_corr = pow(U_corr, (-1) * self.half_v_n)

            U_v = [1.0 + u * u / self.v for u in U.flatten()]
            U_v = np.multiply.reduce(U_v)

            U_v = pow(U_v, (-1) * self.half_items)

            return self.c * self.g * self.g_n * U_corr / U_v

    def calculate_v(self):
        m = -1000000.0
        max_v = 0
        for v in range(self.n_items - 2):
            self.init_v(v + 1)
            s = 0.0
            for row in self.data_subset.values.values:
                try:
                    density = self.calculate_copula_density(row)
                    s += np.log(density)
                except:
                    pass
                if not np.isfinite(s):
                    break
            if s > m and np.isfinite(s):
                m = s
                max_v = v + 1
        return max_v

    def calculate_t_correlation_matrix(self):
        cov_matrix = np.zeros((self.n_features, self.n_features), dtype=float)
        for row in self.data_subset.values.values:
            sigma = self.calculate_features_distribution(row)
            sigma = stats.t.ppf(sigma, self.v).reshape(1, -1)
            cov_matrix += sigma.T @ sigma
        cov_matrix = (1.0 / self.n_items) * cov_matrix
        diagonal = np.diag(1.0 / np.sqrt(cov_matrix.diagonal()))
        return diagonal @ cov_matrix @ diagonal
