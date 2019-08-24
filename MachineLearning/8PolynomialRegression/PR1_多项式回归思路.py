import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
多项式回归思路(手工实现)
'''

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)        #-3到3随机数列向量
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

print("=============使用线性回归(效果不好)====================")
lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_predict=lin_reg.predict(X)
print("score: ",lin_reg.score(X,y))

# plt.scatter(x,y)
# plt.plot(x,y_predict,color='r')
# plt.show()

print("=============解决方案,添加一个特征====================")
print("原本X: ",X.shape)  #原本的数据集只有一个特征
X2=np.hstack([X,X**2])
print("X加上一个特征X^2: ",X2.shape)  #新的数据集每个样本有两个特征

lin_reg2=LinearRegression()
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)
print("score: ",lin_reg2.score(X2,y))

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
plt.show()

'''y = 0.5 * x**2 + x + 2'''
print("x和x^2前面的系数: ",lin_reg2.coef_)
print("截距: ",lin_reg2.intercept_)
