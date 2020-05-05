""" Description
Writea subroutine “convert(X)”to convert thecategorical features in data instances X to one-hot encoding representations.
(For example, data instances have one feature named “body-temperature”, then we should convert “warm-blooded”and“cold-blooded”to [1,0] and[0,1] respectively.)
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
def onehot_convert(X):
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X).toarray().tolist()
    return X

def z_score_convert(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X).tolist()
    return X    
    
if __name__ == "__main__":
    conv = onehot_convert([['warm-blooded'],['cold-blooded']])
    print('test one_hot:\n', conv)
    conv = z_score_convert([[-2], [2]])
    print('test z score:\n', conv)