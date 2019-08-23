from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

"""
网格搜索(调库实现)(寻找好的超参数)
使用sklearn封装的Grid Search进行网格搜索
"""

digits=datasets.load_digits()                   #手写数字库
X=digits.data                                   #1797 * 64 数据集
y=digits.target                                 #1791 * 1  特征集

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)   #分割训练和测试数据集

print("==============网格搜索=================")
param_grid=[
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]

knn_clf=KNeighborsClassifier()
grid_search=GridSearchCV(knn_clf,param_grid,n_jobs=-1)    #参数：n_jobs=-1 用多少个核进行计算(-1用全部核); verbose=2 运行时输出过程
grid_search.fit(X_train,y_train)                #耗时操作

print(grid_search.best_estimator_)              #详细结果
'''
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=1, p=4,
           weights='distance')
'''
print()
print(grid_search.best_params_)                 #结果
'''{'n_neighbors': 3, 'p': 2, 'weights': 'distance'}'''
print()
print(grid_search.best_score_)                  #分类准确度

knn_clf=grid_search.best_estimator_             #使用网格搜索的结果作为最终算法
print(knn_clf.score(X_test,y_test))             #分类准确度


