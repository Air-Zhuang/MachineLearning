import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
多元线性回归(调库)
'''

boston=datasets.load_boston()               #波士顿房产数据
x=boston.data                               #(506, 13) 13个特征值
y=boston.target                             #(506,) 房价
x=x[y<50.0]                                 #去掉y=最大房价的所有极端值
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=666)

print("=============调用sklearn的线性回归======================")
reg=LinearRegression()
reg.fit(X_train,y_train)
print("系数:",reg.coef_)
print("截距:",reg.intercept_)
print("score:",reg.score(X_test,y_test))

print("=============解释模型======================")
print("特征影响从小到大排名(绝对值): ",boston.feature_names[np.argsort(reg.coef_)])
