import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')

'''
sklearn中使用线性SVM(调库实现)
使用线性SVM的思路解决分类问题,线性SVM不可以传入核函数
默认使用L2模型正则化
默认使用ovr做多分类任务
'''

iris=datasets.load_iris()
X=iris.data
y=iris.target
X=X[y<2,:2]         #暂时只处理二分类问题，方便可视化，只取前两个特征
y=y[y<2]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

standardScaler=StandardScaler()     #SVM需要归一化处理
standardScaler.fit(X_train)
X_train_standard=standardScaler.transform(X_train)
X_test_standard=standardScaler.transform(X_test)


def plot_svc_decision_boundary(model, axis):
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

    w = model.coef_[0]
    b = model.intercept_[0]

    # w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]

    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')

print("================sklearn中使用线性SVM解决二分类任务===========================")
'''
默认使用模型正则化，需要传入超参数C(C越大，容错空间越小)
C越大越趋向于Hard Margin SVM
'''

'''Hard Margin SVM'''
svc=LinearSVC(C=1e9)        #Hard Margin SVM
svc.fit(X_train_standard,y_train)
print("分类准确度: ",svc.score(X_test_standard,y_test))
y_log_predict=svc.predict(X_test_standard)
print("混淆矩阵: \n",confusion_matrix(y_test, y_log_predict))
print("精准率: ",precision_score(y_test, y_log_predict))
print("召回率: ",recall_score(y_test, y_log_predict))
print("F1 Score: ",f1_score(y_test, y_log_predict))

plot_svc_decision_boundary(svc,axis=[-3,3,-3,3])
plt.scatter(X_train_standard[y_train==0,0],X_train_standard[y_train==0,1])
plt.scatter(X_train_standard[y_train==1,0],X_train_standard[y_train==1,1])
plt.show()

'''Soft Margin SVM'''
svc2=LinearSVC(C=0.01)      #Soft Margin SVM
svc2.fit(X_train_standard,y_train)
plot_svc_decision_boundary(svc2,axis=[-3,3,-3,3])
plt.scatter(X_train_standard[y_train==0,0],X_train_standard[y_train==0,1])
plt.scatter(X_train_standard[y_train==1,0],X_train_standard[y_train==1,1])
plt.show()

