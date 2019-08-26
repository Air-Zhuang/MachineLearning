import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.simplefilter('ignore')

'''
sklearn中使用随机森林(调库实现)
Extra-Trees_极其随机森林(调库实现)
随机森林也可以解决回归问题，用的时候请百度
'''

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=666)

plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

print("===============sklearn中使用随机森林====================")
'''
n_estimators=500                模型中有500颗决策树
oob_score=True                  采用放回模式最终取不到的样本做测试/验证
n_jobs=-1                       使用全部核计算
'''
rf_clf=RandomForestClassifier(n_estimators=500,
                              random_state=666,
                              oob_score=True,
                              n_jobs=-1)

rf_clf.fit(X,y)
print("随机森林oob分类准确度: ",rf_clf.oob_score_)


'''可以使用决策树中的超参数例如max_leaf_nodes'''
rf_clf2=RandomForestClassifier(n_estimators=500,
                               max_leaf_nodes=16,
                              random_state=666,
                              oob_score=True,
                              n_jobs=-1)

rf_clf2.fit(X,y)
print("随机森林oob分类准确度: ",rf_clf2.oob_score_)

print("===============sklearn中使用Extra-Trees====================")
'''极其随机森林的决策树在节点划分上，使用随机的特征和随机的阈值'''

'''
bootstrap=True                  放回取样
'''
et_clf=ExtraTreesClassifier(n_estimators=500,
                            bootstrap=True,
                            oob_score=True,
                            random_state=666)


et_clf.fit(X,y)
print("Extra-Trees oob分类准确度: ",et_clf.oob_score_)

print("==================随机森林解决回归问题======================")
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
