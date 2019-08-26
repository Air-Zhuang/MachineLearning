import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

'''
衡量回归算法的标准(手工/调库实现)
    MSE     均方误差
    RMSE    均方根误差
    MAE     平均绝对误差
    R Squared
'''

boston=datasets.load_boston()               #波士顿房产数据
x=boston.data[:,5]                          #(506,) 只使用房间数量(RM)这个特征
y=boston.target                             #(506,) 房价
print("房价中最大值: ",np.max(y))
x=x[y<50.0]                                 #去掉y=最大房价的所有极端值
y=y[y<50.0]

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=666)
X_train=np.array([X_train]).T               #使用简单线性回归要传入列向量
X_test=np.array([X_test]).T
y_train=np.array([y_train]).T
y_test=np.array([y_test]).T

print("=============使用简单线性回归====================")
reg=LinearRegression()
reg.fit(X_train,y_train)
print("a: ",reg.coef_[0][0])
print("b: ",reg.intercept_[0])


plt.scatter(X_train,y_train)
plt.plot(X_train,reg.predict(X_train),color='r')
# plt.show()

y_predict=reg.predict(X_test)

print("=============MSE均方误差====================")
mse_test=np.sum((y_predict-y_test)**2 / len(y_test))
print("MSE: ",mse_test)

print("=============RMSE均方根误差====================")
rmse_test=sqrt(mse_test)
print("RMSE: ",rmse_test)

print("=============MAE平均绝对误差====================")
mae_test=np.sum(np.absolute(y_predict-y_test) / len(y_test))
print("MAE: ",mae_test)

print("=============R Squared====================")
RSquared=1-mean_squared_error(y_test,y_predict) / np.var(y_test)
print("RSquared: ",RSquared)

print("=============sklearn中的MSE,MAE,R Squared====================")
print("MSE: ",mean_squared_error(y_test,y_predict))
print("MAE: ",mean_absolute_error(y_test,y_predict))
print("RSquared: ",r2_score(y_test,y_predict))
print("score: ",reg.score(X_test,y_test))       #score实际使用了r2_score