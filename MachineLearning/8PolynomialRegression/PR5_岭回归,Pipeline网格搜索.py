import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

'''
岭回归(调库实现)

常用,但计算量大
岭回归是模型正则化的一种方式
岭回归可以优化多项式回归的过拟合问题
sklearn中岭回归封装了线性回归，就是在线性回归LinearRegression的基础上加上了L2正则项
正则化是在参数学习中使用限制参数达到防止过拟合的目的的。正则化的思路不适用于非参数学习。所以knn和决策树都不适用于正则化方法
'''

'''
如何在将Pipeline使用网格搜索(调库实现)
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

print("==============岭回归解决多项式回归问题=======================")
def RidgeRegression(degree=2, alpha=1):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),    #多项式的增加特征
        ("std_scaler", StandardScaler()),               #归一化
        ("ridge_reg", Ridge(alpha=alpha))               #岭回归替代了线性回归
    ])

ridge1_reg=RidgeRegression(20,0.0001)
ridge1_reg.fit(X_train,y_train)
y1_predict=ridge1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(ridge1_reg)

ridge1_reg=RidgeRegression(20,3)
ridge1_reg.fit(X_train,y_train)
y1_predict=ridge1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(ridge1_reg)

ridge1_reg=RidgeRegression(20,10000)
ridge1_reg.fit(X_train,y_train)
y1_predict=ridge1_reg.predict(X_test)
print("MSE: ",mean_squared_error(y_test,y1_predict))
plot_model(ridge1_reg)


print("==============如何在将Pipeline使用网格搜索=======================")
ridge_reg=RidgeRegression()

param_grid = [
    {
        'poly__degree': [i for i in range(1, 11)],      #格式: Pipe的名字__Pipe的参数
        'ridge_reg__alpha': [i for i in range(1,6)]
    }
]
grid_search = GridSearchCV(ridge_reg, param_grid,n_jobs=-1)
grid_search.fit(X_train,y_train)

print("网格搜索分类准确率: ",grid_search.best_score_)
print("网格搜索最佳超参数组合: ",grid_search.best_params_)
