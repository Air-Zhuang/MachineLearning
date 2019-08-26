import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

import warnings
warnings.simplefilter('ignore')

'''
Bagging集成学习(调库实现)
不使用多种机器学习模型组合的模式，而使用单一机器学习模型，训练多个模型的方式
oob:Bagging采用放回模式最终取不到的样本做测试/验证
'''

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

print("=================使用Bagging集成学习=======================")
'''
DecisionTreeClassifier()        这里使用决策树这种非参数的学习方式更能产生出差异相对比较大的子模型
n_estimators=500                集成500个决策树模型
max_samples=100                 每个子模型看100个样本数据
bootstrap=True                  使用放回取样的方式
n_jobs=-1                       使用全部核计算
max_features=100                每次对100个特征进行随机取样,这时可以不用max_samples(用在特征数量多的情况下)
'''
bagging_clf=BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=500,
                              max_samples=100,
                              bootstrap=True,n_jobs=-1)

bagging_clf.fit(X_train,y_train)
print("Bagging集成学习分类准确度: ",bagging_clf.score(X_test, y_test))

print("=================Bagging集成学习使用oob=======================")
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

bagging_clf=BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=500,
                              max_samples=100,
                              bootstrap=True,n_jobs=-1,oob_score=True)

bagging_clf.fit(X,y)
print("Bagging集成学习使用oob分类准确度: ",bagging_clf.oob_score_)