import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
线性回归算法中使用批量梯度下降(手工实现)
'''

np.random.seed(666)
x=2*np.random.random(size=100)
y=x*3.+4+np.random.normal(size=100)     #(100,)
X=x.reshape(-1,1)                       #(100,1)

plt.scatter(x,y)
plt.show()

print("==============使用批量梯度下降法训练====================")
def J(theta,X_b,y):         #损失函数的值
    return np.sum((y-X_b.dot(theta))**2) / len(X_b)

def dJ(theta,X_b,y):        #求导
    return X_b.T.dot(X_b.dot(theta)-y) / len(X_b)


def gradient_descent(X_b, y, initial_theta, eta,epsilon=1e-8):      #批量梯度下降
    theta = initial_theta
    while True:
        gradient = dJ(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
    return theta

X_b = np.hstack([np.ones((len(x), 1)), x.reshape(-1,1)])            #先将x处理成x_b格式
initial_theta = np.zeros(X_b.shape[1])                              #(100, 2)
eta = 0.01                                                          #(2,)     [0. 0.]

theta=gradient_descent(X_b,y,initial_theta,eta)
print(theta)                                #[截距 斜率] [b a]


print("==============LinearRegression使用批量梯度下降法====================")
lin_reg=LinearRegression()
lin_reg.fit(X,y)                            #sklearn的fit其实用的就是梯度下降法
print(lin_reg.coef_)
print(lin_reg.intercept_)