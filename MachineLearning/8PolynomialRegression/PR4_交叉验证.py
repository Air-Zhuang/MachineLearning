from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=1978)

'''
交叉验证(调库实现)
sklearn的网格搜索GridSearchCV已经默认使用了交叉验证
'''

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=666)

print("================交叉验证=========================")
knn_clf=KNeighborsClassifier()
'''
自动进行交叉验证过程，返回生成的k(默认3)个模型，每个模型对应的准确率
'''
print(cross_val_score(knn_clf,X_train,y_train,cv=3))


# '''手工进行网格搜索'''
# best_k, best_p, best_score = 0, 0, 0
# for k in range(2, 11):
#     for p in range(1, 6):
#         knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
#         # knn_clf.fit(X_train, y_train)
#         # score = knn_clf.score(X_test, y_test)               #只使用训练数据，测试数据拟合的模型分类准确度
#         scores = cross_val_score(knn_clf, X_train, y_train)
#         score = np.mean(scores)                             #使用交叉验证拟合的三个模型的平均分类准确率
#         if score > best_score:
#             best_k, best_p, best_score = k, p, score
#
# print("Best K =", best_k)
# print("Best P =", best_p)
# print("Best Score =", best_score)
#
# '''使用网格搜索出的最佳超参数创建模型'''
# best_knn_clf=KNeighborsClassifier(weights="distance",n_neighbors=2,p=2)
# best_knn_clf.fit(X_train,y_train)
# print(best_knn_clf.score(X_train,y_train))

print("===========网格搜索GridSearchCV已经默认使用了交叉验证=================")
param_grid=[
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(2,11)],
        'p':[i for i in range(1,6)]
    }
]

grid_search=GridSearchCV(knn_clf,param_grid,verbose=1,n_jobs=-1,cv=3)
'''
Fitting 3 folds for each of 45 candidates, totalling 135 fits
解读:在这个网格搜索中，每一次会使用交叉验证的方式，把训练数据集分成三份，
     现在的参数组合有45中可能，每组参数要分成三份进行交叉验证计算平均值，
     总共要进行135次训练
'''
grid_search.fit(X_train,y_train)
print("网格搜索分类准确率: ",grid_search.best_score_)
print("网格搜索最佳超参数组合: ",grid_search.best_params_)

best_knn_clf=grid_search.best_estimator_
print("最终模型分类准确率: ",best_knn_clf.score(X_test,y_test))

