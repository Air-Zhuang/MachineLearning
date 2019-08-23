import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

'''
拆分 训练数据集,测试数据集(手工实现)
'''

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成训练数据集和测试数据集"""
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ration must be valid"

    if seed:
        np.random.seed(seed)                                #如果需要随机数种子

    shuffled_indexes = np.random.permutation(len(X))        #相当于shuffle操作,可以用于矩阵

    test_size = int(len(X) * test_ratio)                    #拆分比例，测试数据集大小
    test_indexes = shuffled_indexes[:test_size]             #测试数据集索引
    train_indexes = shuffled_indexes[test_size:]            #训练数据集索引

    X_train = X[train_indexes]                              #使用fancy-index取出测试数据集和训练数据
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test

print("==================使用自己的分离训练，测试数据集的函数测试KNN准确率===================")
iris=datasets.load_iris()       #加载鸢尾花数据集

X=iris.data                     #150 * 4 数据集
Y=iris.target                   #150 * 1 特征集

X_train, X_test, y_train, y_test=train_test_split(X,Y)
print("X_train: ",X_train.shape)
print("y_train: ",y_train.shape)
print("X_test: ",X_test.shape)
print("y_test: ",y_test.shape)
print()

my_knn_clf=KNeighborsClassifier(n_neighbors=3)
my_knn_clf.fit(X_train,y_train)
y_predict=my_knn_clf.predict(X_test)                #传入测试数据集
print(y_predict)                                    #X_test的训练结果
print(y_test)                                       #测试集的实际结果

print(sum(y_predict==y_test)/len(y_test))           #求出分类准确度
