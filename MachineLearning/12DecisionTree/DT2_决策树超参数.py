import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.simplefilter('ignore')

'''
决策树的超参数(调库实现)
'''

X,y=datasets.make_moons(noise=0.25,random_state=666)

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

print("==============决策树的超参数===================")
'''
max_depth=2                     决策树最高深度为2
min_samples_split=10            对于一个节点，至少要有10个数据，才对这个节点继续拆分下去
min_samples_leaf=6              对于一个叶子节点，至少有6个数据
max_leaf_nodes=4                最多有4个叶子节点
'''

dt_clf=DecisionTreeClassifier()     #过拟合
dt_clf.fit(X,y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

dt_clf=DecisionTreeClassifier(max_depth=2)
dt_clf.fit(X,y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

dt_clf=DecisionTreeClassifier(min_samples_split=10)
dt_clf.fit(X,y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

dt_clf=DecisionTreeClassifier(min_samples_leaf=6)
dt_clf.fit(X,y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

dt_clf=DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf.fit(X,y)
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()