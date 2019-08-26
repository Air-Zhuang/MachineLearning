import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')

'''
SVM使用多项式核函数(调库实现)
常用高斯核函数
'''

X,y=datasets.make_moons(noise=0.15,random_state=666)   #(100, 2)(100,)
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

print("====================SVM使用多项式核函数=========================")
'''
SVC
kernel="poly"       设置核函数为多项式核函数
degree=3            多项式核函数的超参数d(阶数)为3
coef0=0.0           多项式核函数的超参数c,默认为0.0
C=1.0               默认使用模型正则化，需要传入超参数C(C越大，容错空间越小),C越大越趋向于Hard Margin SVM
'''
def PolynomialKernelSVC(degree,C=1.0):
    return Pipeline([
        ("std_scaler",StandardScaler()),
        ("kernelSVC",SVC(kernel="poly",degree=degree,coef0=0.0,C=C))
    ])

poly_kernel_svc=PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X_train,y_train)
print("分类准确度: ",poly_kernel_svc.score(X_test,y_test))
y_log_predict=poly_kernel_svc.predict(X_test)
print("混淆矩阵: \n",confusion_matrix(y_test, y_log_predict))
print("精准率: ",precision_score(y_test, y_log_predict))
print("召回率: ",recall_score(y_test, y_log_predict))
print("F1 Score: ",f1_score(y_test, y_log_predict))

plot_decision_boundary(poly_kernel_svc,axis=[-1.5,2.5,-1.0,1.5])
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1])
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1])
plt.show()
