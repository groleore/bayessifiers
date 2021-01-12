import numpy as np
import math

from .abstract import AbstractBayesClassifier, AbstractClassPredictor


class GaussianNaiveBayesClassifier(AbstractBayesClassifier):
    
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)

    def get_class_predictor(self, data, **kwargs):
        return GaussianNaiveClassPredictor(data, **kwargs)

    def get_vars(self):
        var = [predictor.var for predictor in self.class_predictors]
        return np.stack(var, axis=0)

    def get_vars(self):
        var = [predictor.var for predictor in self.class_predictors]
        return np.stack(var, axis=0)
    
    @classmethod
    def get_name(cls):
        return 'Gaussian Naive Bayes'


class GaussianNaiveClassPredictor(AbstractClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)
        
        calculator = self.data_subset.get_calculator()
        self.mean = calculator.calculate_mean()
        self.var = calculator.calculate_var()

    def calculate_aposteriory_prob(self, X):
        prob = 1.0
        for x, mean, var in zip(X, self.mean, self.var):
            prob *= self.calculate_aposteriory_feature_prob(x, mean, var)
        return prob    
    
    def calculate_aposteriory_feature_prob(self, X, mean, var):
        pi_coef = math.sqrt(2 * math.pi * var)
        c = 1.0 / pi_coef
        
        X_centered = X - mean
        exp_degree = X_centered ** 2 / (2 * var)
        exp = math.exp(-1.0 * exp_degree)
        
        return c * exp


class CustomNaiveBayesClassifier(AbstractBayesClassifier):
    
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, **kwargs)
    
    def get_class_predictor(self, data, **kwargs):
        return GaussianNaiveClassPredictor(data, **kwargs)
    
    def init_class_predictors(self, **kwargs):
        return [CustomNaiveBayesClassPredictor(d) for d in self.data_splitter.splitted_data]


class CustomNaiveBayesClassPredictor(AbstractClassPredictor):
    
    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)
        self.distributions = self.data_subset.get_distributions()
    
    def calculate_aposteriory_prob(self, X):
        density = [d_e.pdf(x) for x, d_e in zip(X, self.distributions)]
        with np.errstate(invalid='raise'):
            try:
                return np.multiply.reduce(density)
            except:
                return 1e-20
