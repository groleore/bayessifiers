import numpy as np
import math
from scipy import stats
from collections.abc import Iterable
from abc import ABC, abstractproperty, abstractmethod

class AbstractDistribution(ABC):

    def __init__(self, X, **kwargs):
        self.X = X
        self.eps = 1e-50

    @abstractmethod
    def pdf(self, x):
        return 

    @abstractmethod
    def cdf(self, x):
        return

    def ks_test(self):
        return stats.ks_1samp(self.X, self.cdf)

    @classmethod
    def get_name(cls):
        pass

class AbstractParametricDistribution(AbstractDistribution):

    def __init__(self, X, **kwargs):
        super().__init__(X, **kwargs)
        [*params] = self.distribution_class.fit(X)
        self.params = params
        self.distribution = self.distribution_class(*params)

    def pdf(self, x):
        return self.distribution.pdf(x)

    def cdf(self, x):
        return self.distribution.cdf(x)

    @abstractproperty
    def distribution_class(self):
        pass

    @classmethod
    def get_name(cls):
        pass


class DistributionEstimator():

    def __init__(self, X):
        self.X = X
        self.param_distributions = {d.get_name(): d for d in AbstractParametricDistribution.__subclasses__()}

    def get_distribution(self, **kwargs):
        distribution = kwargs.get('dist', 'kernel')
        param_distribution = self.param_distributions.get(distribution, None)
        if param_distribution:
            return param_distribution(self.X, **kwargs)
        if distribution == 'fit':
            return self.fit_distribution(**kwargs)
        else:
            return KernelEstimatedDistribution(self.X, **kwargs)

    def fit_distribution(self, **kwargs):
        max_pvalue = 0.0
        best_distribution = None
        for distribution_obj in self.param_distributions.values():
            try:
                distribution = distribution_obj(self.X, **kwargs)
                statistic, pvalue = distribution.ks_test()
                if pvalue > max_pvalue:
                    max_pvalue = pvalue
                    best_distribution = distribution
            except:
                pass

        if max_pvalue > 0.05:
            return best_distribution
        else:
            return KernelEstimatedDistribution(self.X, **kwargs)


class NormalDistribution(AbstractParametricDistribution):
    @property
    def distribution_class(self):
        return stats.norm

    @classmethod
    def get_name(cls):
        return 'norm'

class LognormDistribution(AbstractParametricDistribution):
    @property
    def distribution_class(self):
        return stats.lognorm

    @classmethod
    def get_name(cls):
        return 'lognorm'


class GammaDistribution(AbstractParametricDistribution):
    @property
    def distribution_class(self):
        return stats.gamma

    @classmethod
    def get_name(cls):
        return 'gamma'


class RayleighDistribution(AbstractParametricDistribution):
    @property
    def distribution_class(self):
        return stats.rayleigh

    @classmethod
    def get_name(cls):
        return 'rayleigh'


class GumbelDistribution(AbstractParametricDistribution):
    @property
    def distribution_class(self):
        return stats.gumbel_r

    @classmethod
    def get_name(cls):
        return 'gumbel_r'


KERNELS = dict()


def kernel_function(func):
    if isinstance(func, staticmethod):
        KERNELS[func.__func__.__name__] = func.__func__
    else:
        KERNELS[func.__name__] = func
    return func


class KernelEstimatedDistribution(AbstractDistribution):

    def __init__(self, X, **kwargs):
        super().__init__(X, **kwargs)

        self.m = len(X)
        self.X_h = kwargs.get('X_h', [])
        if len(self.X_h) == 0:
            self.X_h = X

        kernel = kwargs.get('kernel', 'epanechnikov')
        h_arg = kwargs.get('h', 'skott')
        self.h = self.calculate_h(h_arg)

        self.kernel = KERNELS.get(kernel, None)
        if self.kernel == None:
            self.kernel = KERNELS['epanechnikov']

        self.L = min(X)
        self.R = max(X)
        self.integrator = NumericalIntegrator(self.pdf, self.h, self.L)
        self.cache = {}
    
    def calculate_h(self, h):
        std = np.std(self.X_h)
        m = len(self.X_h)

        if type(h) == int or type(h) == float:
            return h
        elif h == 'silverman':
            return std * pow(0.75 * m, -0.2)
        else:
            # Skott method
            return std * pow(m, -0.2)

    @kernel_function
    @staticmethod
    def epanechnikov(x):
        if abs(x) > 1.0:
            return 0.0
        c = 3.0 / 4.0
        r = 1.0 - x * x
        return c * r

    @kernel_function
    @staticmethod
    def quartic(x):
        if abs(x) > 1.0:
            return 0.0
        c = 15.0 / 16.0
        r = 1.0 - x * x
        return c * r * r

    @kernel_function
    @staticmethod
    def triangle(x):
        if abs(x) > 1.0:
            return 0.0
        return 1.0 - abs(x)

    @kernel_function
    @staticmethod
    def rectangle(x):
        if abs(x) > 1.0:
            return 0.0
        return 0.5

    @kernel_function
    @staticmethod
    def cosinus(x):
        if abs(x) > 1.0:
            return 0.0
        return (math.pi / 4.0) * math.cos(math.pi / 2.0 * x)

    @kernel_function
    @staticmethod
    def gaussian(x):
        c = math.pow(math.pi * 2.0, -1.0 / 2.0)
        p = np.exp(-0.5 * x * x)
        return c * p

    def pdf(self, x):
        cache_result = self.cache.get(x, False)
        if cache_result:
            return cache_result
        else:
            kernel_values = [self.kernel((x - x_i) / self.h) for x_i in self.X]
            c = self.h * self.m
            result = np.sum(kernel_values) / c
            if result == 0.0:
                result = self.eps
            self.cache[x] = result
            return result

    def cdf(self, X):
        if isinstance(X, Iterable):
            return [self.cdf(x) for x in X]
        if X < self.L:
            return 0.0
        if X > self.R:
            return 1.0
        return self.integrator.calculate(X)


class NumericalIntegrator():

    def __init__(self, f, eps, L):
        self.f = f
        self.eps = eps
        self.L = L - 5 * eps

    def calculate(self, R):
        L = self.L
        D = np.arange(L + self.eps, R - self.eps, self.eps)
        result = 0.0
        for d in D:
            result += self.f(d)

        result += (self.f(L) + self.f(R)) / 2.0
        S = R - L
        c = S / (len(D) + 1)
        result = result * c

        return result
