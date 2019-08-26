import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('ignore')

'''
SVM解决回归问题(调库实现)
'''

boston=datasets.load_boston()
X=boston.data
y=boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

print("================SVM使用线性SVR解决回归问题=========================")
def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("linearSVR",LinearSVR(epsilon=epsilon,C=1.0))
    ])

svr=StandardLinearSVR()
svr.fit(X_train,y_train)
print("R Squared: ",svr.score(X_test,y_test))