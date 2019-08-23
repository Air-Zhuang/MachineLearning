import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets


print("======================sklearn_KNN=======================================================")
raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x_train=np.array(raw_data_x)    #训练使用的x和y
y_train=np.array(raw_data_y)
'''
调用sklearn中的kNN算法
'''
x = np.array([[8.093607318, 3.365731514]])                  #预测这个点
kNN_classifier=KNeighborsClassifier(n_neighbors=6)          #创建kNN实例，传入k的值
kNN_classifier.fit(x_train,y_train)                         #拟合(训练)操作，这一步有返回值
x_predict=kNN_classifier.predict(x)                         #这里即使只预测一个数据也要转换成二维矩阵
print(x_predict)

print("====================sklearn中的train_test_split===============================================")
iris=datasets.load_iris()       #加载鸢尾花数据集

X=iris.data                     #150 * 4 数据集
Y=iris.target                   #150 * 1 特征集

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2)        #默认是0.2，可以传入random_state=666(随机数种子)

knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train,y_train)
y_predict=knn_clf.predict(X_test)                   #传入测试数据集

print(y_predict)                                    #X_test的训练结果
print(y_test)                                       #测试集的实际结果

print(knn_clf.score(X_test,y_test))                 #KNN分类准确度
print(accuracy_score(y_test,y_predict))             #KNN分类准确度
