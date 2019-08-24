import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)

'''
绘制决策边界(调库,手工实现)
'''

iris=datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2,:2]       #(100, 2)由于逻辑回归只能解决二分类问题，从数据集中筛选出0和1分类的鸢尾花，方便可视化，只取前两个特征
y = y[y<2]          #(100,)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)
print("===============逻辑回归(调库实现)=========================")
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print("分类准确度: ",log_reg.score(X_test,y_test))

# print("X_test每个概率值: ",log_reg.predict_proba(X_test))
print("X_test分类结果向量: ",log_reg.predict(X_test))

print("系数: ",log_reg.coef_)
print("截距: ",log_reg.intercept_)

print("===============绘制两个特征值的决策边界=========================")
def x2(x1):     #两个特征值的分界线计算公式
    return (-log_reg.coef_[0][0] * x1 - log_reg.intercept_) / log_reg.coef_[0][1]

x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)

plt.scatter(X[y==0,0], X[y==0,1], color="red")
plt.scatter(X[y==1,0], X[y==1,1], color="blue")
plt.plot(x1_plot, x2_plot)
plt.show()


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


plot_decision_boundary(log_reg, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()

print("===============绘制两个特征的KNN的决策边界=========================")
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
print("分类准确度: ",knn_clf.score(X_test, y_test))

plot_decision_boundary(knn_clf, axis=[4, 7.5, 1.5, 4.5])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

print("===============绘制三个特征的KNN的决策边界=========================")
knn_clf_all = KNeighborsClassifier(n_neighbors=50)  #k越大，决策边界越规整
knn_clf_all.fit(iris.data[:,:2], iris.target)
print("分类准确度: ",knn_clf.score(X_test, y_test))

plot_decision_boundary(knn_clf_all, axis=[4, 8, 1.5, 4.5])
plt.scatter(iris.data[iris.target==0,0], iris.data[iris.target==0,1])
plt.scatter(iris.data[iris.target==1,0], iris.data[iris.target==1,1])
plt.scatter(iris.data[iris.target==2,0], iris.data[iris.target==2,1])
plt.show()

print("===============绘制带多项式的逻辑回归决策边界=========================")
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array((X[:,0]**2+X[:,1]**2)<1.5, dtype='int')

def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X, y)
print("分类准确度: ",poly_log_reg.score(X, y))

plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()