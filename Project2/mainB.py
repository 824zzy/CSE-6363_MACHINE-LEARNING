""" Description
Write a computer program to implement KNNmethod.
(i) the program canrepresent each data instance ---these are “stored data”.
(ii) When a new data instance is presented, the program compares the new data instance to every “stored data instances”to compute the distance, and find out the k nearest neighbors, and predict the classlabel for the new data instance.
"""
import math
class KNN:
    def __init__(self, K):
        self.K = K
        self.distances = []
        self.k_neighbors = []
    
    @staticmethod
    def distance(X1, X2):
        dist = 0
        for (x1, x2) in zip(X1, X2):
            dist += (x1 - x2) ** 2
        return dist
    
    def fit_predict(self, X_train, y_train, test_sample):
        self.k_neighbors, self.distances = [], []
        for X, y in zip(X_train, y_train):
            d = self.distance(X, test_sample)
            self.distances.append((X, y, d))
        self.distances.sort(key=lambda x: x[-1]) # sort by distance
        self.k_neighbors = [sample[0:-1] for sample in self.distances[0:self.K]]
        
        label_votes={}
        for neighbor in self.k_neighbors:
            label = neighbor[-1]
            if label in label_votes.keys():
                label_votes[label] += 1
            else:
                label_votes[label] = 1
        sorted_votes=sorted(label_votes.items(), key=lambda kv: kv[1], reverse=True) ## sorted by vote numbers
        return sorted_votes[0][0]
        
    
if __name__ == "__main__":
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    knn = KNN(3)
    pred = knn.fit_predict(X, y, [1.1])
    print("Predicted class is: ", pred)