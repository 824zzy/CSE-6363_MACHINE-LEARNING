"""
w of SVM is:
 [-1.53459653 -0.66683961]
b of SVM is:
 [8.83787556]
The prediction value of each data instance is:
 [ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
 -1 -1 -1 -1 -1 -1 -1 -1]
The indexes of support cectors is:
 [100 122 123 131 149 171  26  46  57  68  81  84]
alpha of SVM is:
 -2.3012978087309515
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
import matplotlib.pyplot as plt

def dataGenerator(seed, n):
    '''
    Randomly generating two data distrbution
    '''
    n1 = n
    mu1 = np.array([2, 3])
    sigma1 = 2.5 * np.matrix([[1, -1], [-1, 3]])

    np.random.seed(seed)
    x1 = np.random.multivariate_normal(mu1, sigma1, n1).T
    y1 = np.ones([1, n1], dtype=np.int64)

    n2 = n
    mu2 = np.array([6, 6])
    sigma2 = 2.5 * np.matrix([[1, -1], [-1, 3]])
    np.random.seed(seed + 100)
    x2 = np.random.multivariate_normal(mu2, sigma2, n2).T
    y2 = -1 * np.ones([1, n2], dtype=np.int64)

    x = np.concatenate((x1, x2), axis=1)
    y = np.concatenate((y1, y2), axis=1)

    return x, y

if __name__ == "__main__":
    X, y = dataGenerator(1001778274, 100)
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X.T, y.T)
    w = clf.coef_[0]
    alpha = -w[0]/w[1]
    b = clf.intercept_
    preds = clf.predict(X.T)
    
    
    print('w of SVM is:\n', w)
    print('b of SVM is:\n', b)
    print('The prediction value of each data instance is:\n', preds)
    print('The indexes of support cectors is:\n', clf.support_)
    print('alpha of SVM is:\n', alpha)
