from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.simplefilter('ignore')

'''
决策树解决回归问题(调库实现)
DecisionTreeRegressor和DecisionTreeClassifier的超参数相同，不同的是输出结果是一个值
'''

boston=datasets.load_boston()
x=boston.data
y=boston.target
X_train, X_test, y_train, y_test=train_test_split(x,y,random_state=666)

print("================决策树解决回归问题(非常容易产生过拟合现象)========================")
dt_reg=DecisionTreeRegressor()
dt_reg.fit(X_train,y_train)

print("R Squared: ",dt_reg.score(X_test,y_test))