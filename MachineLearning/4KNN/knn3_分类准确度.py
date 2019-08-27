import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

'''
分类准确度(调库实现)
    解决分类问题中使用
'''

digits=datasets.load_digits()           #手写数字库
X=digits.data                           #1797 * 64 数据集
y=digits.target                         #1791 * 1  特征集

some_digit_image=X[666].reshape(8,8)                    #将一个手写数据绘图
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary)  #cmap只是将像素由彩色变成黑白
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)   #分割训练和测试数据集

print("==============求分类准确度=================")
knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train,y_train)
y_predict=knn_clf.predict(X_test)
print(sum(y_predict==y_test)/len(y_test))       #分类准确度

print(accuracy_score(y_test,y_predict))         #sklearn分类准确度
print(knn_clf.score(X_test,y_test))             #sklearn分类准确度

