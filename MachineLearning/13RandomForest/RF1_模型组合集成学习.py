import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

import warnings
warnings.simplefilter('ignore')

'''
sklearn中使用集成学习(调库实现)
使用多种机器学习模型组合的模式
'''

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)

print("逻辑回归分类准确度: ",log_clf.score(X_test, y_test))

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print("SVM分类准确度: ",svm_clf.score(X_test, y_test))

dt_clf = DecisionTreeClassifier(random_state=666)
dt_clf.fit(X_train, y_train)
print("决策树分类准确度: ",dt_clf.score(X_test, y_test))

print("=================sklearn中使用集成学习=======================")
'''
voting='hard'           以少数服从多数的方式决定最终结果
voting='soft'           以加权平均的方式决定最终结果
'''
voting_clf = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],voting='hard')

voting_clf.fit(X_train, y_train)
print("集成学习分类准确度: ",voting_clf.score(X_test, y_test))

voting_clf2 = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=666))],voting='soft')

voting_clf2.fit(X_train, y_train)
print("集成学习分类准确度: ",voting_clf2.score(X_test, y_test))
