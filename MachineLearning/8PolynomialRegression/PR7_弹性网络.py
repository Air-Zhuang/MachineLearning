import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet

'''
弹性网络(调库实现)

结合了岭回归和LASSO的优点，但是新增了一个超参数，在实际中先用岭回归
弹性网络是模型正则化的一种方式
弹性网络可以优化多项式回归的过拟合问题
sklearn中弹性网络封装了线性回归
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

print("==============弹性网络解决多项式回归问题=======================")
def ElasticnetRegression(degree, l1_ratio, alpha):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("EN_reg", ElasticNet(l1_ratio=l1_ratio, alpha=alpha))
    ])


EN_reg = ElasticnetRegression(20, 0.3, 0.01)
EN_reg.fit(X_train, y_train)
y1_predict = EN_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(EN_reg)
