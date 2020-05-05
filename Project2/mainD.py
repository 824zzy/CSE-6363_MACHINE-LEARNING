""" Description
Write a computer program to implement Centroidmethod.(i) the program can represent all the data instances as “k centroids”(assumingthe number of classes is k) ---these are “stored data”.  (ii) When a new data instance is presented, the program compares the new data instance to “k stored centroids”to compute the distance, and find out the nearest neighbor, and predict the class label for the new data instance.
"""
from mainC import onehot_convert, z_score_convert
import numpy as np

def centroid_method(X_train, y_train, X_test, y_test):
    labels = np.unique(y_train)
    centroids = np.zeros((X_train.shape[0], len(labels)))
    y_output = []

    for l in range(len(labels)):
        centroids[:,l] = np.mean(X_train[:, y_train == labels[l]], axis=1)

    for xt in np.transpose(X_test):
        distance = np.linalg.norm(centroids - np.vstack(xt), axis=0)
        y_output.append(labels[np.argmin(distance)])

    score = np.mean(np.array(y_output) == y_test)
    return score
        
if __name__ == "__main__":
    filename = "../Project3/data/HandWrittenLetters.txt"
    dataset = np.loadtxt(filename, delimiter=",")
    print(dataset.shape)
    X, Y = dataset[1:], dataset[0]
    print(X.shape, Y.shape)
    X_train, X_test = X[:, :800], X[:, 800:]
    y_train, y_test = Y[:800], Y[800:]
    score = centroid_method(X_train, y_train, X_test, y_test)
    print(score)