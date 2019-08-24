import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
sklearn中使用多项式回归(调库实现)
sklearn没有提供多项式回归的类，我们可以用Pipeline组装一个
'''

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)                #-3到3随机数列向量
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

print("============sklearn中使用多项式回归======================")
poly=PolynomialFeatures(degree=2)    #为原本的数据集添加最多几次幂相应的特征
poly.fit(X)
X2=poly.transform(X)
'''
    (100, 3)
    [ 1.         -0.6435811   0.41419663]
    x^0          x^1          x^2
'''
print(X2.shape)         #(100, 3)

lin_reg2=LinearRegression()
lin_reg2.fit(X2,y)
y_predict2=lin_reg2.predict(X2)
print("score: ",lin_reg2.score(X2,y))
print("系数:x^0,x^1,x^2: ",lin_reg2.coef_)
print("截距: ",lin_reg2.intercept_)

# plt.scatter(x,y)
# plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
# plt.show()

print("===============关于PolynomialFeatures==============================")
'''
sklearn中的PolynomialFeatures会生成所有小于等于degree相应的所有多项式的项
随着degree增长，样本总的特征数成指数级增长
'''
X=np.arange(1,11).reshape(-1,2)     #(5, 2) (x,y)
# print(X)

poly=PolynomialFeatures(degree=2)
poly.fit(X)
X2=poly.transform(X)                #(5, 6) (1,x,y,x^2,x*y,y^2)
# print(X2)

print("===============Pipeline==============================")
'''
由于多项式回归需要进行多项式的特征，归一化，线性回归三步
Pipeline可以将这三步合在一起，使我们每一次调用的时候不需要重复这三步
'''
np.random.seed(667)
x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x**2 + x + 2 + np.random.normal(0, 1, 100)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),     #多项式的增加特征
    ("std_scaler", StandardScaler()),           #归一化
    ("lin_reg", LinearRegression())             #线性回归
])

poly_reg.fit(X_train,y_train)
y_predict=poly_reg.predict(X_test)
print("score: ",poly_reg.score(X_test,y_test))
print("MSE: ",mean_squared_error(y_test,y_predict))
