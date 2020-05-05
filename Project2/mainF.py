"""Description
Writea computerprogramto preprocess the given dataset “american_people_1000.txt”. 
Data file contains 15 attributes and last attribute “Income”is the class label. 
(i) Following attributes: “fnlwgt”, “EducationNum”, “CapitalGain”, “CapitalLoss” will not be used. 
You should drop those columns.
(ii) Partition the dataset into training set (1st-900th instances) and testing set (901st-1000th instances). 
(iii) Train
your implementedKNN, Centroid, and SVM classifiers on thosetraining data instances, and report their classification accuracyon testing data instances.(iv) Check if SVM has different classification performancesusing linear/Gaussiankernel.
"""
import pandas as pd
import numpy as np 
from mainB import KNN
# from mainD import 
from mainC import onehot_convert, z_score_convert
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from mainD import centroid_method
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.set_printoptions(threshold=np.inf)

# Read data and drop unused columns
dataset = pd.read_csv('./data/american_people_1000.txt', sep=' ')
dataset = dataset.drop(['fnlwgt', 'EducationNum', 'CapitalGain', 'CapitalLoss'], axis=1)
# Preprocess the dataset
dataset = onehot_convert(dataset)
# Partition the dataset
X_train, y_train = [d[:-1] for d in dataset[:900]], [d[-1] for d in dataset[:900]]
X_test, y_test = [d[:-1] for d in dataset[900:]], [d[-1] for d in dataset[900:]]
# KNN
knn = KNN(3)
preds = []
for i in tqdm(range(len(X_test))):
    preds.append(knn.fit_predict(X_train, y_train, X_test[i]))
acc = accuracy_score(y_test, preds)
print("The accuracy of KNN:", acc)
# Centroid Method
# print(np.array(X_train).shape.T, np.array(X_test).shape.T, np.array(y_train).shape, np.array(y_test).shape)
score = centroid_method(np.array(X_train).T, np.array(y_train), np.array(X_test).T, np.array(y_test))
print("The accuracy of centroid method is:", score)
## SVM: Linaer/Gaussian
# ------ Linear -----
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# pred = clf.predict(X_test)
# acc = accuracy_score(y_test, pred)
# print("The accuracy of SVM:", acc)
# ------ Linear -----
# ------ Gaussian -----
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("The accuracy of SVM:", acc)
# ------ Gaussian -----
