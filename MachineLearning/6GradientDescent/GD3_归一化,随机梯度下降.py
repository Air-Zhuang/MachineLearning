from sklearn import datasets
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
梯度下降之前需要归一化(调库实现)
    不归一化会导致每一次移动的距离或者很大或者很小
'''

'''
随机梯度下降(调库实现)
'''

boston=datasets.load_boston()               #波士顿房产数据
x=boston.data                               #(506, 13) 13个特征值
y=boston.target                             #(506,) 房价
x=x[y<50.0]                                 #去掉y=最大房价的所有极端值
y=y[y<50.0]
X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=666)

print("=============归一化处理======================")
standardScaler=StandardScaler()
standardScaler.fit(X_train)
X_train_standard=standardScaler.transform(X_train)
X_test_standard=standardScaler.transform(X_test)

print("=============LinearRegression使用批量梯度下降法======================")
reg=LinearRegression()
reg.fit(X_train_standard,y_train)
print("系数: ",reg.coef_)
print("截距:",reg.intercept_)
print("score:",reg.score(X_test_standard,y_test))

print("=============SGDRegressor使用随机梯度下降法======================")
reg=SGDRegressor(n_iter_no_change=5)  #n_iter_no_change:整个样本浏览多少次,默认5
reg.fit(X_train_standard,y_train)
print("系数: ",reg.coef_)
print("截距:",reg.intercept_)
print("score:",reg.score(X_test_standard,y_test))

