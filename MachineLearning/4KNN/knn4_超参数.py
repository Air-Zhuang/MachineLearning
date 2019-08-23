from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
超参数(手工实现)
超参数1(n_neighbors)：K个数
超参数2(weights)：是否考虑距离
超参数3(p)：明可夫斯基距离的p
'''

digits=datasets.load_digits()           #手写数字库
X=digits.data                           #1797 * 64 数据集
y=digits.target                         #1791 * 1  特征集

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)   #分割训练和测试数据集

print("==============寻找最好的k=================")
best_score=0
best_k=None

for k in range(1,11):
    knn_clf=KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train,y_train)
    score=knn_clf.score(X_test,y_test)          #分类准确度
    if score>best_score:
        best_k=k
        best_score=score

print("best_k: ",best_k)
print("best_score: ",best_score)

print("==============考虑距离作为权重=================")
best_method=None
best_score=0
best_k=None

for method in ["uniform","distance"]:       #uniform：不考虑距离  distance：考虑距离
    for k in range(1,11):
        knn_clf=KNeighborsClassifier(n_neighbors=k,weights=method)
        knn_clf.fit(X_train,y_train)
        score=knn_clf.score(X_test,y_test)          #分类准确度
        if score>best_score:
            best_k=k
            best_score=score
            best_method=method

print("best_k: ",best_k)
print("best_score: ",best_score)
print("best_method: ",best_method)

print("==============搜索明可夫斯基距离的p(耗时)=================")
best_p=None
best_score=0
best_k=None

for k in range(1,11):
    for p in range(1,6):
        knn_clf=KNeighborsClassifier(n_neighbors=k,weights="distance",p=p)
        knn_clf.fit(X_train,y_train)
        score=knn_clf.score(X_test,y_test)          #分类准确度
        if score>best_score:
            best_k=k
            best_score=score
            best_p=p

print("best_k: ",best_k)
print("best_score: ",best_score)
print("best_p: ",best_p)