import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)

'''
sklearn中使用逻辑回归(调库实现)
sklearn中逻辑回归默认使用模型正则化,模型为L2,超参数C的值为1.0
'''

np.random.seed(666)
X=np.random.normal(0,1,size=(200,2))
y=np.array(X[:,0]**2+X[:,1]<1.5,dtype='int')
for _ in range(20):
    y[np.random.randint(200)]=1
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

def plot_decision_boundary(model, axis):
    '''专业绘制决策边界'''
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

print("===============sklearn中使用逻辑回归=========================")

print("----------使用默认的简单线性回归---------")
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print("逻辑回归默认使用的模型正则化: ",log_reg.penalty)
print("默认使用的模型正则化的超参数C: ",log_reg.C)
print("分类准确度: ",log_reg.score(X_test, y_test))

print("----------使用多项式回归-------------")
def PolynomialLogisticRegression(degree,C=1.0,penalty='l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=C,penalty=penalty))
    ])


poly_log_reg = PolynomialLogisticRegression(degree=2)   #使用sklearn中默认配置
poly_log_reg.fit(X, y)
print("分类准确度: ",poly_log_reg.score(X_test, y_test))
plot_decision_boundary(poly_log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

poly_log_reg = PolynomialLogisticRegression(degree=20,C=0.1)   #degree过拟合,靠C拉回来
poly_log_reg.fit(X, y)
print("分类准确度: ",poly_log_reg.score(X_test, y_test))
plot_decision_boundary(poly_log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

poly_log_reg = PolynomialLogisticRegression(degree=20,C=0.1,penalty='l1')   #使用l1
poly_log_reg.fit(X, y)
print("分类准确度: ",poly_log_reg.score(X_test, y_test))
plot_decision_boundary(poly_log_reg,axis=[-4,4,-4,4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()