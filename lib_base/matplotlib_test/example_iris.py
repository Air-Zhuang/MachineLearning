import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

print("==============读取数据和简单的数据探索=================")

iris=datasets.load_iris()                                           #加载鸢尾花的数据集
iris.DESCR                                                          #数据集描述
iris.data                                                           #数据集合
iris.data.shape                                                     #数据集形状
iris.feature_names                                                  #数据集每列代表什么
iris.target                                                         #数据集的每条样本代表的鸢尾花类型(0,1,2)
iris.target_names                                                   #鸢尾花类型(['setosa' 'versicolor' 'virginica'])

print("==============针对前两个维度绘图=================")
X=iris.data[:,:2]                                                   #取数据集的前两列
y=iris.target                                                       #数据集的每条样本代表的鸢尾花类型(0,1,2)

plt.scatter(X[y==0,0],X[y==0,1],color="red",label="type 0")         #从样本X中取花型为0的第一列数据作为x轴，花型为0的第二列作为y轴
plt.scatter(X[y==1,0],X[y==1,1],color="blue",label="type 1")        #从样本X中取花型为1的第一列数据作为x轴，花型为1的第二列作为y轴
plt.scatter(X[y==2,0],X[y==2,1],color="green",label="type 2")       #从样本X中取花型为2的第一列数据作为x轴，花型为2的第二列作为y轴
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()


print("==============针对后两个维度绘图=================")
X=iris.data[:,2:]                                                   #取数据集的后两列
y=iris.target                                                       #数据集的每条样本代表的鸢尾花类型(0,1,2)

plt.scatter(X[y==0,0],X[y==0,1],color="red",label="type 0")
plt.scatter(X[y==1,0],X[y==1,1],color="blue",label="type 1")
plt.scatter(X[y==2,0],X[y==2,1],color="green",label="type 2")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.legend()
plt.show()