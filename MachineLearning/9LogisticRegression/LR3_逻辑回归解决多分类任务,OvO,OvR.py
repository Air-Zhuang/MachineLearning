import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=469)

'''
逻辑回归解决多分类任务(调库实现)
使用OvR和OvO
OvO更耗时,但是结果更准确
sklearn的逻辑回归默认使用OvR进行多分类任务
'''
'''
使用sklearn独立封装的OvO和OvR可以将其他的二分类任务改成多分类任务(调库实现)
'''

iris=datasets.load_iris()
X=iris.data[:,:2]       #方便可视化，只取前两个特征
y=iris.target

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

print("============逻辑回归解决多分类任务===================")
'''使用ovr'''
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print("默认使用ovr: ",log_reg.multi_class)
print(log_reg.score(X_test,y_test))

plot_decision_boundary(log_reg, axis=[4, 8.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

'''使用ovo'''
log_reg2=LogisticRegression(multi_class='multinomial',solver='newton-cg')   #使用ovo需要这样设置
log_reg2.fit(X_train,y_train)
print("使用ovo: ",log_reg2.multi_class)
print(log_reg2.score(X_test,y_test))

plot_decision_boundary(log_reg2, axis=[4, 8.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.scatter(X[y==2,0], X[y==2,1])
plt.show()

print("============分类全部鸢尾花数据集===================")
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

'''使用ovr'''
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print(log_reg.score(X_test,y_test))

'''使用ovo'''
log_reg2=LogisticRegression(multi_class='multinomial',solver='newton-cg')   #使用ovo需要这样设置
log_reg2.fit(X_train,y_train)
print(log_reg2.score(X_test,y_test))

print("============分类手写digits数据集===================")
digits=datasets.load_digits()
X=digits.data
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

'''使用ovr'''
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print(log_reg.score(X_test,y_test))

'''使用ovo'''
log_reg2=LogisticRegression(multi_class='multinomial',solver='newton-cg')   #使用ovo需要这样设置
log_reg2.fit(X_train,y_train)
print(log_reg2.score(X_test,y_test))

print("================使用sklearn独立封装的OvO和OvR可以将其他的二分类任务改成多分类任务==============")
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

ovr=OneVsRestClassifier(log_reg)
ovr.fit(X_train,y_train)
print(ovr.score(X_test,y_test))

ovo=OneVsOneClassifier(log_reg)
ovo.fit(X_train,y_train)
print(ovo.score(X_test,y_test))