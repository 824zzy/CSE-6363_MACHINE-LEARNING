""" Description
Write a computer program to implement KNNmethod.
(i) the program canrepresent each data instance ---these are “stored data”.
(ii) When a new data instance is presented, the program compares the new data instance to every “stored data instances”to compute the distance, and find out the k nearest neighbors, and predict the classlabel for the new data instance.
"""
import math
class KNN:
    def __init__(self, train_set, K):
        self.train_set = train_set
        self.K = K
        self.distance = []
        self.dist = 0
        self.classes = list(set([c[-1] for c in train_set]))
    
    def classify(self, test_set):
        pred = []
        for test_case in test_set:
            for row in self.train_set:
                for x, y in zip(row[:-1], test_case):
                    self.dist += (x-y)**2
                self.distance.append(row+[math.sqrt(self.dist)])
                self.dist = 0
            self.distance.sort(key=lambda x: x[-1])
            neightbors = self.distance[:self.K]
            
            res = [0] * len(self.classes)
            for case in neightbors:
                for idx, c in enumerate(self.classes):
                    if case[-2] == c:
                       max(enumerate(res), key=lambda x: x[1]) res[c] += 1
            print(max(enumerate(res), key=lambda x: x[1]))
            pred.append(max(enumerate(res), key=lambda x: x[1]))
            self.distance = []
        return pred
            
            
            
if __name__ == "__main__":
    knn = KNN()