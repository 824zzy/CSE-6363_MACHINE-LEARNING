"""Description
Write a computer program to implement SVM method.
(i)preprocessthe data instances:if data instances have categorical features, then call the subroutine convert(X). 
(ii) given the training data instances, your program should be able to compute the w, b, alpha, margin.
(iii) when a new data instance is presented, the program uses pre-trained SVM model to predict the class label for the new data instance.
"""
from sklearn import svm
import matplotlib.pyplot as plt 
import numpy as np

def plot_decision_function(classifier, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')
    # axis.axis('off')
    axis.set_title(title)
    
if __name__ == "__main__":
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)
    # pred = clf.predict([[2., 2.]])
    print(clf.dual_coef_)
    w = clf.coef_[0]
    alpha = -w[0]/w[1]
    b = clf.intercept_
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    print('w of SVM is:', w)
    print('b of SVM is:', b)
    print('alpha of SVM is:', alpha)
    print('margin of SVM is:', margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_function(clf, axes[0], "Constant weights")
    plt.show()
