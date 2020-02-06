import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import itertools
import functools
import numpy as np
import pylab
from sklearn.metrics import mean_squared_error

def plot(X_train, y_train, X_true, y_true):
    plt.scatter(X_train, y_train, label='train data')
    plt.plot(X_true, y_true, c='g', label='ground truth')
    plt.title('Figure 1.2')
    plt.ylabel('Y axis ')
    plt.xlabel('X axis')
    plt.legend()
    plt.show()

def genDummy(start, end, nums, std):
    X = np.linspace(start, end, nums)
    sinusoid = np.sin(X*2*np.pi)
    noise = np.random.normal(scale=std,size=nums)
    y = sinusoid+noise
    return X, y

def rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))

class FeatureFormation():
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()
    
class LinearRegression():
    def fit(self, X:np.ndarray, t:np.ndarray):
        self.w = np.linalg.pinv(X)@t
        self.var = np.mean(np.square(X@self.w-t))

    def predict(self, X:np.ndarray, return_std:bool=False):
        y = X @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y
    
class RidgeRegression():
    def __init__(self, alpha:float=1.):
        self.alpha = alpha

    def fit(self, X, t):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)

    def predict(self, X):
        return X @ self.w