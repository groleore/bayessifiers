import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

class ModelValidator():
    
    def __init__(self, model_cls, data, **kwargs):
        X, y = data
        self.X = X
        self.y = y
        self.model_cls = model_cls
        self.kwargs = kwargs
    
    def plot_desicion_areas(self, h=0.04, feature_indexes=[0, 1], ):
        X = self.X.iloc[:, feature_indexes]
        X_columns = self.X.columns[0:2]
        classifier = self.model_cls(X, self.y, **self.kwargs)

        X = X.values[:, 0:2]
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = X1.min() - 0.5, X1.max() + 0.5
        X2_min, X2_max = X2.min() - 0.5, X2.max() + 0.5
        xx_1, xx_2 = np.meshgrid(np.arange(X1_min, X1_max, h),
                                 np.arange(X2_min, X2_max, h))

        Z = classifier.predict(np.c_[xx_1.ravel(), xx_2.ravel()])

        Z = Z.reshape(xx_1.shape)
        plt.figure(figsize=(8, 6), dpi=80,)
        plt.pcolormesh(xx_1, xx_2, Z, cmap=cmap_light, shading='auto')

        plt.scatter(X1, X2, c=self.y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx_1.min(), xx_1.max())
        plt.ylim(xx_2.min(), xx_2.max())
        plt.xlabel(X_columns[0])
        plt.ylabel(X_columns[1])
        plt.show()
        
    def estimate_model_quality(self):
        random_states = [1, 13, 42, 517, 68]
        test = []
        train = []
        for state in random_states:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=state)
            model = self.model_cls(X_train, y_train, **self.kwargs)
            train.append(model.score(X_train, y_train))
            test.append(model.score(X_test, y_test))
        return np.mean(train), np.std(train), np.mean(test), np.std(test)
