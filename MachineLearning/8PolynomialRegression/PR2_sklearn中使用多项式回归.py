import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

'''
sklearn中使用多项式回归(调库实现)
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
print("系数:x^0,x^1,x^2: ",lin_reg2.coef_)
print("截距: ",lin_reg2.intercept_)

# plt.scatter(x,y)
# plt.plot(np.sort(x),y_predict2[np.argsort(x)],color='r')
# plt.show()

print("===============关于PolynomialFeatures==============================")
X=np.arange(1,11).reshape(-1,2)     #(5, 2)
