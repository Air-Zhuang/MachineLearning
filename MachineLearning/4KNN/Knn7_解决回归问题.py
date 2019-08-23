from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

'''
使用sklearn对KNN封装的Regressor解决回归问题
'''

boston=datasets.load_boston()               #波士顿房产数据
x=boston.data                               #(506, 13) 13个特征值
y=boston.target                             #(506,) 房价
x=x[y<50.0]                                 #去掉y=最大房价的所有极端值
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=666)

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

print("=============KNN的Regressor======================")
knn_reg=KNeighborsRegressor()
grid_search=GridSearchCV(knn_reg,param_grid,n_jobs=-1)

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)         #最好的参数{'n_neighbors': 5, 'p': 1, 'weights': 'distance'}

print("score: ",grid_search.best_estimator_.score(X_test,y_test))