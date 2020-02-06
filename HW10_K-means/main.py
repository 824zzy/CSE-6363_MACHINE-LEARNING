import numpy as np
from misc import *

x1 = np.random.normal(size=(100, 2))
x1 += np.array([5, 5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([-5, 5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([-5, -5])
x_train = np.vstack((x1, x2, x3))
x0, x1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
x = np.array([x0, x1]).reshape(2, -1).T

max_iter = 5
kmeans = KMeans(n_clusters=3)
centers = kmeans.fit(x_train, max_iter)
cluster = kmeans.predict(x_train)

for i in range(max_iter):
    plt.subplot(2, 2, i+1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=cluster)
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=200, marker='X', lw=2, c=['purple', 'cyan', 'yellow'], edgecolor="white")
    plt.contourf(x0, x1, kmeans.predict(x).reshape(100, 100), alpha=0.1)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()