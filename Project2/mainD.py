""" Description
Writea subroutine “convert(X)”to convert thecategorical features in data instances X to one-hot encoding representations.
(For example, data instances have one feature named “body-temperature”, then we should convert “warm-blooded”and“cold-blooded”to [1,0] and[0,1] respectively.)
"""
from sklearn.preprocessing import OneHotEncoder
def convert(X):
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X)
    return X
    
    
    
if __name__ == "__main__":
    conv = convert([['warm-blooded','a'],['cold-blooded', 'a']])
    print(conv)