import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.simplefilter('ignore')

'''
Boosting继承学习(调库实现)
Boosting没有oob
Ada Boosting
Gradient Boosting
Boosting解决回归问题，用的时候自行百度
'''

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=666)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()


print("==============Ada Boosting===================")
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2), n_estimators=500)
ada_clf.fit(X_train, y_train)
print("Ada Boosting分类准确度:",ada_clf.score(X_test, y_test))

print("==============Gradient Boosting===================")
'''使用决策树，无需传入'''
gb_clf = GradientBoostingClassifier(max_depth=2, n_estimators=30)
gb_clf.fit(X_train, y_train)
print("Gradient Boosting分类准确度:",gb_clf.score(X_test, y_test))

print("==============Boosting解决回归问题===================")
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor