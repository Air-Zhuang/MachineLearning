from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

'''
数据归一化(调库实现)
1、最值归一化
2、均值方差归一化(大多数情况用这个!!!!)
'''


print("==============均值方差归一化StandardScaler=================")
iris=datasets.load_iris()
X=iris.data         #(150, 4)
y=iris.target       #(150,)
print("没有做归一化处理: ",X[:2])
print()

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=666)

standardScaler=StandardScaler()
standardScaler.fit(X_train)
print("X_train的均值: ",standardScaler.mean_)
print("X_train的标准差: ",standardScaler.scale_)

X_train_standard=standardScaler.transform(X_train)           #归一化处理
X_test_standard=standardScaler.transform(X_test)

knn_clf=KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train_standard,y_train)
print()
print(knn_clf.score(X_test_standard,y_test))        #分类准确度(传入的测试数据集也必须要经过归一化处理)

