import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

pd.get_dummies()

enc = OneHotEncoder(categories='auto')
enc.fit([[0], [1], [2], [3],[4],[5]])
def onehot(a):
    return enc.transform([a]).toarray()

print(onehot([4]))

# X=np.array([[100000,0.09,5],[200000,0.08,3],[300000,0.07,1],[400000,0.06,2],[500000,0.05,4],[600000,0.04,3],[700000,0.06,5]])
# #
# for i in range(5):
#     print(onehot([i]))
#
# X=np.array([[10,10,0. ,0. ,0. ,0. ,1., 0.],[20,20,0. ,0., 1. ,0. ,0. ,0.],[30,30,1. ,0. ,0.,0. ,0., 0.],[40,40,0. ,1., 0. ,0., 0. ,0.],
#             [50,50,0. ,0. ,0. ,1., 0. ,0.],[60,60,0., 0., 1., 0., 0., 0.],[70,70,0., 0. ,0. ,0., 1. ,0.]])
# # X=np.array([[10,10,5],[20,20,3],[30,30,1],[40,40,2],[50,50,4],[60,60,3],[70,70,5]])
# # X=np.array([[10,10,5],[20,20,3],[30,30,1],[40,40,2],[50,50,4],[60,60,3],[70,70,5]])
# y=np.array([[100],[200],[300],[400],[500],[600],[700]])
# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=666)
#
#
#
# reg=LinearRegression()
# reg.fit(X_train,y_train)
# print("score:",reg.score(X_test,y_test))