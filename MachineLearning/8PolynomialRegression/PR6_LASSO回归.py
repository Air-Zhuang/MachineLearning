import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

'''
LASSO回归(调库实现)

LASSO用的少,更多用于做特征选择
LASSO回归是模型正则化的一种方式
LASSO回归可以优化多项式回归的过拟合问题
sklearn中LASSO回归封装了线性回归，就是在线性回归LinearRegression的基础上加上了L1正则项
正则化是在参数学习中使用限制参数达到防止过拟合的目的的。正则化的思路不适用于非参数学习。所以knn和决策树都不适用于正则化方法
'''

np.random.seed(42)
x=np.random.uniform(-3.0,3.0,size=100)
X=x.reshape(-1,1)
y=0.5*x+3+np.random.normal(0,1,size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

def plot_model(model):          #封装画图工具
    X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(X_plot)
    plt.scatter(x, y)
    plt.plot(X_plot[:,0], y_plot, color='r')
    plt.axis([-3, 3, 0, 6])
    plt.show()

print("==============LASSO回归解决多项式回归问题=======================")
def LassoRegression(degree=2, alpha=1):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),    #多项式的增加特征
        ("std_scaler", StandardScaler()),               #归一化
        ("lasso_reg", Lasso(alpha=alpha))               #LASSO回归替代了线性回归
    ])

lasso1_reg=LassoRegression(20,0.01)
lasso1_reg.fit(X_train,y_train)
y1_predict=lasso1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(lasso1_reg)

lasso1_reg=LassoRegression(20,0.1)
lasso1_reg.fit(X_train,y_train)
y1_predict=lasso1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(lasso1_reg)

lasso1_reg=LassoRegression(20,1)
lasso1_reg.fit(X_train,y_train)
y1_predict=lasso1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(lasso1_reg)