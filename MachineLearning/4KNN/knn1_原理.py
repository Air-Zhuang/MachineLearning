import numpy as np
import matplotlib.pyplot as plt

'''
KNN原理(手工实现)
'''

raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x_train=np.array(raw_data_x)    #训练使用的x和y
y_train=np.array(raw_data_y)

print(x_train[y_train==0,0])
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],color="g",label="type 0")
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],color="r",label="type 1")
plt.legend()
# plt.show()

print("==============KNN实现过程=================")
x = np.array([8.093607318, 3.365731514])                        #预测这个点

from math import sqrt
distances=[sqrt(np.sum((x_train - x)**2))for x_train in x_train]        #用原生方法求所有x_train的欧拉距离
print("distances: ",distances)                                  #distances的索引相当于x_train的索引

nearest=np.argsort(distances)                                   #distances排序索引(相当于对x_train排序)
print("nearest: ",nearest)
k=6
topK_y=[y_train[i] for i in nearest[:k]]                        #根据前6个索引从y_train中取得标签
print("topK_y: ",topK_y)

from collections import Counter
votes=Counter(topK_y)
print(votes)
print(votes.most_common(1))
predict_y=votes.most_common(1)[0][0]
print(predict_y)                                                #取最多的投票结果对应的标签

print("==============按照sklearn的规则封装自己的kNN算法=================")
class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
                "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
                "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
