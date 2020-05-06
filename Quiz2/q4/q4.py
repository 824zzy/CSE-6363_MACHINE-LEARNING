"""
   Age   G  GS    MP   FG  FGA    FG%   3P  3PA    3P%   2P  2PA    2P%  \
0   27  77  77  33.7  4.6  6.6  0.703  0.0  0.0  0.000  4.6  6.6  0.704   
1   32  26   3  17.2  2.7  6.2  0.441  1.7  4.4  0.377  1.1  1.8  0.596   
2   25  18   2   5.7  0.7  2.2  0.325  0.1  0.3  0.200  0.7  1.9  0.343   
3   23  66   0  18.6  3.5  7.7  0.451  0.8  2.7  0.309  2.6  5.0  0.529   
4   28  64  24  22.9  4.0  9.9  0.409  2.2  6.0  0.367  1.8  3.9  0.474   

    eFG%   FT  FTA    FT%  ORB   DRB   TRB  AST  STL  BLK  TOV   PF  PS/G  
0  0.703  3.5  8.0  0.430  3.5  10.3  13.8  1.2  0.7  2.3  1.4  2.7  12.7  
1  0.575  0.4  0.5  0.786  0.5   3.1   3.6  1.0  0.3  0.3  0.5  2.2   7.5  
2  0.338  0.4  0.4  0.875  0.2   1.3   1.6  0.3  0.1  0.2  0.2  1.1   1.9  
3  0.506  1.0  1.3  0.727  0.7   3.2   3.8  1.2  0.6  0.5  0.8  1.6   8.8  
4  0.521  1.5  1.9  0.750  0.4   2.3   2.7  1.0  0.8  0.5  1.1  2.2  11.8  
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 87.21it/s]
The prediction of KNN: ['P', 'P', 'S', 'P', 'P', 'S', 'P', 'P', 'P', 'S', 'P', 'P', 'C', 'C', 'P', 'P', 'P', 'S', 'S', 'S']
The prediction of centroid method is: ['P', 'P', 'P', 'P', 'P', 'S', 'C', 'S', 'C', 'P', 'S', 'P', 'C', 'P', 'C', 'P', 'C', 'P', 'S', 'S']
The prediction of SVM: ['P' 'P' 'P' 'P' 'P' 'S' 'C' 'S' 'P' 'P' 'S' 'P' 'C' 'P' 'C' 'P' 'P' 'S'
 'S' 'S']
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

# Step1: select data file and drop unused player
dataset = pd.read_csv("./Classification_in_NBA_data/NBA_4.txt", sep=' ')
Y = dataset['Pos']
dataset = dataset.drop(['Player', 'Tm', 'Pos'], axis=1)
print(dataset.head())

def minmax_convert(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)

def onehot_convert(X):
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X).toarray().tolist()
    return X

# Partition the dataset
dataset = minmax_convert(dataset)
X_train, y_train = [d[1:] for d in dataset[:482]], [d[0] for d in Y[:482]]
X_test = [d[1:] for d in dataset[482:]]
csv_data = pd.DataFrame(dataset)
csv_data.to_csv('./output.csv', index=False)
# KNN
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

knn = KNN(3)
pred_knn = []
for i in tqdm(range(len(X_test))):
    pred_knn.append(knn.fit_predict(X_train, y_train, X_test[i]))
print("The prediction of KNN:", pred_knn)
# Centroid Method

def centroid_method(X_train, y_train, X_test):
    labels = np.unique(y_train)
    centroids = np.zeros((X_train.shape[0], len(labels)))
    y_output = []

    for l in range(len(labels)):
        centroids[:,l] = np.mean(X_train[:, y_train == labels[l]], axis=1)

    for xt in np.transpose(X_test):
        distance = np.linalg.norm(centroids - np.vstack(xt), axis=0)
        y_output.append(labels[np.argmin(distance)])

    return y_output

pred_cm = centroid_method(np.array(X_train).T, np.array(y_train), np.array(X_test).T)
print("The prediction of centroid method is:", pred_cm)
## SVM: Linaer/Gaussian
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
pred_svm = clf.predict(X_test)
print("The prediction of SVM:", pred_svm)