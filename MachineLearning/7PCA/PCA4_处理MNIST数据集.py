import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata

'''
处理MNIST数据集(调库实现)(需要准备数据)
'''

mnist = fetch_mldata('MNIST original',data_home='E:\workspace3\数据集\datasets')   #~\datasets\mldata\mnist-original.mat

X,y=mnist['data'],mnist['target']
X_train=np.array(X[:60000],dtype=float)     #(60000, 784)   MNIST规定前60000就是训练数据集
y_train=np.array(y[:60000],dtype=float)     #(60000,)
X_test=np.array(X[60000:],dtype=float)      #(10000, 784)
y_test=np.array(y[60000:],dtype=float)      #(10000,)
'''MINST数据集可以不用做归一化处理'''

# knn_clf=KNeighborsClassifier()                        #不做PCA处理的话784维要训练十五分钟
# knn_clf.fit(X_train,y_train)
# print("PCA处理之前knn训练的得分: ",knn_clf.score(X_test,y_test))

pca=PCA(0.9)                                            #保留90%的特征
pca.fit(X_train)
print("经过PCA训练后降到多少维: ",pca.n_components_)     #87维保留了90%的特征

X_train_reduction=pca.transform(X_train)
X_test_reduction=pca.transform(X_test)

knn_clf=KNeighborsClassifier()                          #经过PCA处理的话87维只要训练两分钟
knn_clf.fit(X_train_reduction,y_train)
'''
PCA的过程不仅仅是降维,而且在降维的过程中对原有数据集的噪音消除了,
使我们更加好的拿到数据集的特征,所以识别准确率反而上升
'''
print("PCA处理之前knn训练的得分: ",knn_clf.score(X_test_reduction,y_test))   #0.9728
