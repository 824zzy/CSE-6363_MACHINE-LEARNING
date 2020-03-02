import pandas as pd
import numpy as np

data = pd.read_csv('./vertebrate.txt', sep=' ')

class Naive_Bayes:
    def __init__(self, data):
        self.d = data.iloc[:, 1:]
        self.headers = self.d.columns.values.tolist()
        self.prior = np.zeros(len(self.d['Class'].unique()))
        self.conditional = {}
    
    def build(self):
        y_unique = self.d['Class'].unique()
        for i in range(0,len(y_unique)):
            self.prior[i]=(sum(self.d['Class']==y_unique[i])+1)/(len(self.d['Class'])+len(y_unique))
            
        for h in self.headers[:-1]:
            x_unique = list(set(self.d[h]))
            x_conditional = np.zeros((len(self.d['Class'].unique()),len(set(self.d[h]))))
            for j in range(0,len(y_unique)):
                for k in range(0,len(x_unique)):
                    x_conditional[j,k]=(self.d.loc[(self.d[h]==x_unique[k])&(self.d['Class']==y_unique[j]),].shape[0]+1)/(sum(self.d['Class']==y_unique[j])+len(x_unique))
        
            x_conditional = pd.DataFrame(x_conditional,columns=x_unique,index=y_unique)   
            self.conditional[h] = x_conditional       
        return self.prior, self.conditional
    
    def predict(self, X):
        res = []
        for i in range(len(self.prior)):
            p_i = self.prior[i]
            for j, h in enumerate(self.headers[:-1]):
                p_i *= self.conditional[h][X[j]][i]*self.conditional[h][X[j]][i]
            res.append(p_i)
        return res
    
nb = Naive_Bayes(data)
prior_probability,conditional_probability = nb.build()

# print(prior_probability)
# for k, v in conditional_probability.items():
#     print(k, v)

test_data = ['warm-blooded', 'hair', 'yes', 'no', 'no', 'yes', 'no']
ans = nb.predict(test_data)
print(ans)