import numpy as np

from abc import ABC, abstractmethod

from .data_splitter import DatasetSplitter


class AbstractBayesClassifier(ABC):

    def __init__(self, X, y, **kwargs):
        self.data_splitter = DatasetSplitter(X, y, **kwargs)
        self.class_predictors = self.init_class_predictors(**kwargs)

    def init_class_predictors(self, **kwargs):
        return [self.get_class_predictor(data, **kwargs) for data in self.data_splitter.splitted_data]

    @abstractmethod
    def get_class_predictor(self, data, **kwargs):
        return

    def _calculate_prob(self, X):
        try:
            probs = np.array([predictor.calculate_prob(X) for predictor in self.get_class_predictors()])
            return probs / np.sum(probs)
        except:
            probs = np.array([predictor.calculate_apriori_prob() for predictor in self.get_class_predictors()])
            return probs / np.sum(probs)

    def _predict(self, X):
        probs = self._calculate_prob(X)
        return self.data_splitter.labels[np.argmax(probs)]

    def predict_proba(self, X):
        return self.apply_along_rows(X, self._calculate_prob)

    def get_class_predictors(self):
        return self.class_predictors

    def predict(self, X):
        return self.apply_along_rows(X, self._predict)

    def apply_along_rows(self, X, func):
        X_array = np.array(X)
        if X_array.ndim == 1:
            X_array = X_array[np.newaxis, :]
        return np.apply_along_axis(func, 1, X_array)

    def score(self, X, y):
        prediction = self.predict(X)
        y_array = np.array(y)
        return (prediction == y_array).sum() / len(X)

    def get_means(self):
        means = [predictor.mean for predictor in self.class_predictors]
        return np.stack(means, axis=0)


class AbstractClassPredictor(ABC):

    def __init__(self, labeled_data, **kwargs):
        self.data_subset = labeled_data

    def calculate_apriori_prob(self):
        return self.data_subset.class_prob

    def calculate_prob(self, X):
        apriori_prob = self.calculate_apriori_prob()
        aposteriory_prob = self.calculate_aposteriory_prob(X)
        try:
            return apriori_prob * aposteriory_prob
        except Exception as e:
            return 1e-40

    @abstractmethod
    def calculate_aposteriory_prob(self, X):
        return


class AbstractCopulaClassPredictor(AbstractClassPredictor):

    def __init__(self, labeled_data, **kwargs):
        super().__init__(labeled_data, **kwargs)
        self.distributions = self.data_subset.get_distributions()
        
    def calculate_features_density_multiplication(self, X):
        density = [d_e.pdf(x) for x, d_e in zip(X, self.distributions)]
        return np.multiply.reduce(density)
   
    def calculate_aposteriory_prob(self, X):
        try:
            copula_density = self.calculate_copula_density(X)
            features_density = self.calculate_features_density_multiplication(X)
            return features_density * copula_density
        except Exception as e:
            return 1e-25

    def calculate_features_distribution(self, X):
        eps = 2e-5
        U = np.array([d_e.cdf(x) for x, d_e in zip(X, self.distributions)])
        return np.clip(U, eps, 1.0 - eps)

    @abstractmethod
    def calculate_copula_density(self, X):
        return 
