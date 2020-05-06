"""
The converged loss for each trail:
 [3355.958227996451, 3298.8258640732897, 3394.1043394597286, 3393.0948411283784, 3303.528569296072, 3298.5960666132305, 3392.204283241657, 3385.4587595333746, 3270.553428069688, 3374.336084232904]
The best cluster result is:
 [ 5  5  5  2  5  5  3  5  5  5  5  5  4 12  2 12 12  3  4  6  4 12  4  7
  6  2 13 13  3 13  9 13  7  8  8  8  2  8  8  3  8  8  8  8 14  4 14  2
  4  4  3  4  7  4 14  5  9 14  2  9 14  3 14 14 14 14  1 10 10  1  7  7
  1 10 15 10  7 13 14  4  2  4  7  3  4  4 14  7  6  9  6  2  6  7  3  7
 15  6  7  9  9  9  2  9  9  8 13  9 13  9 11 11 11 11 11 11  3 11 11 11
 11  5 14 14  2 14 14  3 14 14 14 14 12  7 12  1 12 12  3 12 15  4 12  4
 12 12  2 12 12  3  9 12 13 12 14 13  6  1  7  6  3  7  6  6  7]
The confusion matrix is:
Clusters  1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
Labels                                                              
1          0   1   1   0   9   0   0   0   0   0   0   0   0   0   0
2          0   1   1   3   1   1   0   0   0   0   0   4   0   0   0
3          0   1   1   1   0   1   2   0   1   0   0   0   4   0   0
4          0   1   1   0   0   0   0   9   0   0   0   0   0   0   0
5          0   1   1   5   0   0   1   0   0   0   0   0   0   3   0
6          0   1   1   0   1   0   0   0   2   0   0   0   0   6   0
7          3   0   0   0   0   0   3   0   0   4   0   0   0   0   1
8          0   1   1   4   0   0   2   0   0   0   0   0   1   2   0
9          0   1   1   0   0   4   3   0   1   0   0   0   0   0   1
10         0   1   0   0   0   0   0   1   7   0   0   0   2   0   0
11         0   0   1   0   0   0   0   0   0   0  10   0   0   0   0
12         0   1   1   0   1   0   0   0   0   0   0   0   0   8   0
13         1   0   1   1   0   0   1   0   0   0   0   6   0   0   1
14         0   1   1   1   0   0   0   0   1   0   0   6   1   0   0
15         1   0   1   0   0   4   3   0   0   0   0   0   1   1   0
The accuracy before permute confusion matrix is: 0.09696969696969697
The permute index of confusion matrix is:
 [3, 6, 9, 8, 14, 2, 10, 4, 15, 13, 11, 5, 1, 12, 7]
Permuted new confusion matrix is:
 Clusters  3   6   9   8   14  2   10  4   15  13  11  5   1   12  7 
Labels                                                              
1          1   0   0   0   0   1   0   0   0   0   0   9   0   0   0
2          1   1   0   0   0   1   0   3   0   0   0   1   0   4   0
3          1   1   1   0   0   1   0   1   0   4   0   0   0   0   2
4          1   0   0   9   0   1   0   0   0   0   0   0   0   0   0
5          1   0   0   0   3   1   0   5   0   0   0   0   0   0   1
6          1   0   2   0   6   1   0   0   0   0   0   1   0   0   0
7          0   0   0   0   0   0   4   0   1   0   0   0   3   0   3
8          1   0   0   0   2   1   0   4   0   1   0   0   0   0   2
9          1   4   1   0   0   1   0   0   1   0   0   0   0   0   3
10         0   0   7   1   0   1   0   0   0   2   0   0   0   0   0
11         1   0   0   0   0   0   0   0   0   0  10   0   0   0   0
12         1   0   0   0   8   1   0   0   0   0   0   1   0   0   0
13         1   0   0   0   0   0   0   1   1   0   0   0   1   6   1
14         1   0   1   0   0   1   0   1   0   1   0   0   0   6   0
15         1   4   0   0   1   0   0   0   0   1   0   0   1   0   3
The accuracy of confusion matrix is 
 0.2909090909090909
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
np.set_printoptions(threshold=np.inf)

dataFrame = pd.read_csv('./data/YaleFace.txt', sep=' ', header=None)
print(dataFrame.shape)
# labels 
label = dataFrame.iloc[0].astype('int32')
# number of classes
num_class = len(label.unique())
# train data
train_set = dataFrame.iloc[1:, :].T
losses = []
preds = []
for i in range(10):
    kmeans = KMeans(n_clusters=num_class, n_init=1, init='random')
    predicted = np.add(kmeans.fit_predict(train_set), 1)
    centroids = kmeans.cluster_centers_

    # print('Iteration {} Loss: {}'.format(i, kmeans.inertia_))
    losses.append(kmeans.inertia_)
    preds.append(predicted)
    
print("The converged loss for each trail:\n", losses)
print("The best cluster result is:\n", preds[losses.index(min(losses))])

predicted = preds[losses.index(min(losses))]
#  data frame for label and predicted
dataFrame = pd.DataFrame({'Labels' : label.squeeze(), 'Clusters' : predicted})
# confusion matrix
ct = pd.crosstab(dataFrame['Labels'], dataFrame['Clusters'])
print('The confusion matrix is:\n{}'.format(ct))
print('The accuracy before permute confusion matrix is: {}'.format(sum(np.diag(ct)) / dataFrame.shape[0]))

# # bipartite graph
B = nx.Graph()
labels = [str(dataFrame['Labels'][i])+'_x' for i in range(len(dataFrame['Labels']))]
preds = [str(dataFrame['Clusters'][i])+'_y' for i in range(len(dataFrame['Clusters']))]


B.add_nodes_from(labels, bipartite=0)
B.add_nodes_from(preds, bipartite=1)
edges = [(x, y) for x, y in zip(labels, preds)] 
B.add_edges_from(edges)
matching = list(nx.max_weight_matching(B))
matching_list = [0] * num_class
for m in list(matching):
    m0 = int(m[0].split('_')[0]) if m[0].split('_')[1]=='x' else int(m[1].split('_')[0])
    m1 = int(m[1].split('_')[0]) if m[1].split('_')[1]=='y' else int(m[0].split('_')[0])
    matching_list[m0-1] = m1
print("The permute index of confusion matrix is:\n", matching_list)

permuted_ct = ct[matching_list]
print('Permuted new confusion matrix is:\n', permuted_ct)
print('The accuracy of confusion matrix is \n', sum(np.diag(permuted_ct))/dataFrame.shape[0])