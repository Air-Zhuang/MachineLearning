import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import warnings
warnings.simplefilter('ignore')

'''
sklearn中的高斯核函数(调库实现)
'''

X,y=datasets.make_moons(noise=0.15,random_state=666)   #(100, 2)(100,)

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

print("=================sklearn中的高斯核函数==========================")
'''
SVC
kernel="rbf"        设置核函数为高斯核函数
gamma=1.0           高斯核函数的超参数gamma,gamma越小,模型复杂度越低,越欠拟合
'''
def RBFKernelSVC(gamma=1.0):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("svc",SVC(kernel="rbf",gamma=gamma))
    ])


svc=RBFKernelSVC(gamma=0.1)         #欠拟合
svc.fit(X,y)
plot_decision_boundary(svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

svc=RBFKernelSVC(gamma=1.0)
svc.fit(X,y)
plot_decision_boundary(svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

svc=RBFKernelSVC(gamma=10)
svc.fit(X,y)
plot_decision_boundary(svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()

svc=RBFKernelSVC(gamma=100)         #过拟合
svc.fit(X,y)
plot_decision_boundary(svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.show()