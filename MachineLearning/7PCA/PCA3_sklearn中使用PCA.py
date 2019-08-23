import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

'''
sklearn中使用PCA(调库实现)
'''

X=np.empty((100,2))
X[:,0]=np.random.uniform(0.,100.,size=100)
X[:,1]=0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

print("=============sklearn中使用PCA(指定降到多少维的用法)========================")
pca=PCA(n_components=1)                             #指定降到n_components维,一般不这么用,实际应用看最后
pca.fit(X)

print("所有主成分w: ",pca.components_)               #求出的所有主成分w

X_reduction=pca.transform(X)                        #对X做降维处理
print("降维后的X: ",X_reduction.shape)

X_restore=pca.inverse_transform(X_reduction)        #恢复原来的维度
print("恢复原来的维度后的X: ",X_restore.shape)

# plt.scatter(X[:,0],X[:,1],color='b',alpha=0.5)      #原始X和降维再恢复后的X散点图
# plt.scatter(X_restore[:,0],X_restore[:,1],color='r',alpha=0.5)
# plt.show()

print("========================================================")
digits=datasets.load_digits()                       #手写数字识别
X=digits.data                                       #(1797, 64)
y=digits.target                                     #(1797,)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=666)

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
print("PCA处理之前knn训练的得分: ",knn_clf.score(X_test,y_test))

print("=============PCA过程可以看到每一个维度的重要程度========================")
pca=PCA(n_components=X_train.shape[1])              #不降维
pca.fit(X_train)
pca_evr=pca.explained_variance_ratio_               #每一个方向轴对应的重要程度

plt.plot([i for i in range(X_train.shape[1])],[np.sum(pca_evr[:i+1]) for i in range(X_train.shape[1])])
plt.show()                                          #前多少个维度对应的总重要程度之和

print("=============sklearn中PCA处理真实数据========================")
pca=PCA(0.95)                                       #0.95表示降维后保留之前95%的特征,一般都这么用
pca.fit(X_train)
print("经过PCA训练后降到多少维: ",pca.n_components_)
X_train_reduction=pca.transform(X_train)            #降维后的X_train
X_test_reduction=pca.transform(X_test)              #降维后的X_test

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train_reduction,y_train)
print("PCA处理之后knn训练的得分: ",knn_clf.score(X_test_reduction,y_test))

print("=============手写数字降到二维时呈现的关系========================")
pca=PCA(n_components=2)
pca.fit(X)
X_reduction=pca.transform(X)
print(X_reduction.shape)

for i in range(10):
    plt.scatter(X_reduction[y==i,0],X_reduction[y==i,1],alpha=0.8)
plt.show()