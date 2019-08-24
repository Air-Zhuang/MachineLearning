import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
绘制学习曲线(手工实现)
'''

np.random.seed(666)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)
print(X_train.shape)        #(75, 1)

print("===============绘制学习曲线==============================")
def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])     #训练数据集的拟合程度
        train_score.append(mean_squared_error(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)           #测试数据集的拟合程度
        test_score.append(mean_squared_error(y_test, y_test_predict))

    #y轴为MSE
    plt.plot([i for i in range(1, len(X_train) + 1)],np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],np.sqrt(test_score), label="test")
    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 4])
    plt.show()

print("===============线性回归学习曲线(欠拟合)==============================")
'''欠拟合状态下MSE会比较大'''
plot_learning_curve(LinearRegression(), X_train, X_test, y_train, y_test)

print("===============多项式回归学习曲线(正常)==============================")
poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),     #多项式的特征
    ("std_scaler", StandardScaler()),           #归一化
    ("lin_reg", LinearRegression())             #线性回归三步
])

poly_reg.fit(X_train,y_train)
plot_learning_curve(poly_reg, X_train, X_test, y_train, y_test)