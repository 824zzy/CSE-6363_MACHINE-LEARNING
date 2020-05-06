"""
filled mean mp value:
[1.2, 1.5, 7.6000000000000005, 3.1, 3.3000000000000003, 3.0, 1.5333333333333332, 1.5333333333333332, 3.266666666666667, 2.8333333333333335]

distance of 3 neareast neighbors for each data instance (top 10 rows):
0: [56.30273499999998, 88.953123, 94.54220500000001]
1: [36.92244, 40.92067399999999, 41.27929599999999]
2: [54.715531000000006, 79.92057199999999, 80.060257]
3: [55.176736999999996, 87.21090300000002, 87.91457500000001]
4: [41.001459000000004, 63.21999999999999, 68.79535]
5: [111.94484300000003, 149.205558, 155.41422699999995]
6: [17.792424, 18.638671000000002, 22.387905999999994]
7: [31.30162200000001, 33.95761099999999, 35.130157000000004]
8: [59.791374000000005, 67.27059999999999, 70.856778]
9: [49.291447999999995, 56.416478, 64.390715]
"""
"""
The distances of three neighbors: [56.30273499999998, 88.953123, 94.54220500000001]
mean mp is: 1.2
The distances of three neighbors: [36.92244, 40.92067399999999, 41.27929599999999]
mean mp is: 1.5
The distances of three neighbors: [54.715531000000006, 79.92057199999999, 80.060257]
mean mp is: 7.6000000000000005
The distances of three neighbors: [55.176736999999996, 87.21090300000002, 87.91457500000001]
mean mp is: 3.1
The distances of three neighbors: [41.001459000000004, 63.21999999999999, 68.79535]
mean mp is: 3.3000000000000003
The distances of three neighbors: [111.94484300000003, 149.205558, 155.41422699999995]
mean mp is: 3.0
The distances of three neighbors: [17.792424, 18.638671000000002, 22.387905999999994]
mean mp is: 1.5333333333333332
The distances of three neighbors: [31.30162200000001, 33.95761099999999, 35.130157000000004]
mean mp is: 1.5333333333333332
The distances of three neighbors: [59.791374000000005, 67.27059999999999, 70.856778]
mean mp is: 3.266666666666667
The distances of three neighbors: [49.291447999999995, 56.416478, 64.390715]
mean mp is: 2.8333333333333335
Mean mp values for missing data are: [1.2, 1.5, 7.6000000000000005, 3.1, 3.3000000000000003, 3.0, 1.5333333333333332, 1.5333333333333332, 3.266666666666667, 2.8333333333333335]
"""
import math
import pandas as pd
class KNN:
    def __init__(self, K):
        self.K = K
        self.distances = []
        self.k_neighbors = []
    
    @staticmethod
    def distance(x1, x2):
        # Input can be numer-set/group categorical attribute
        dist = 0
        for i in range(len(x1)):
            if isinstance(x1[i], (int, float)):
                dist += (x1[i]-x2[i])**2
            else:
                max_len = max(len(x1[i]), len(x2[i])) if len(x1[i])!=len(x2[i]) else len(x1[i])
                w1, w2 = x1[i].ljust(max_len, '0'), x2[i].ljust(max_len, '0')
                dist += sum([1 if w1[j]!=w2[j] else 0 for j in range(max_len)])
        return dist
    
    def fit_predict(self, X_train, y_train, test_sample):
        self.k_neighbors, self.distances = [], []
        for X, y in zip(X_train, y_train):
            d = self.distance(X, test_sample)
            self.distances.append((X, y, d))
        self.distances.sort(key=lambda x: x[-1]) # sort by distance
        self.k_neighbors = [sample[0:-1] for sample in self.distances[0:self.K]]
        dists = [sample[-1] for sample in self.distances[0:self.K]]
        print("The distances of three neighbors:", dists)
        mean_mp = sum([n[0][6] for n in self.k_neighbors])/self.K
        print('mean mp is:', mean_mp)
        return mean_mp    
    
if __name__ == "__main__":
    # Step 1: 1001778274 select data
    dataset = pd.read_csv('./Fill_Missing_Value_in_NBA_data/NBA_4.txt', sep=' ')
    train, test = dataset[10:], dataset[:10]
    # Build dataset
    X_train, y_train = train[[i for i in train.columns if i!='MP']].values.tolist(), train[['MP']].values.tolist()
    X_test, y_test = test[[i for i in test.columns if i!='MP']].values.tolist(), test[['MP']].values.tolist()

    knn = KNN(3)
    pred_mp = []
    for i in range(len(X_test)):
        mean_mp = knn.fit_predict(X_train, y_train, X_test[i])
        pred_mp.append(mean_mp)
    print("Mean mp values for missing data are:", pred_mp)