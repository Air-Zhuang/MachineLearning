import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=432)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=469)

'''
实现混淆矩阵,精准率,召回率,F1 Score,ROC(调库实现)
'''

digits=datasets.load_digits()
X=digits.data
y=digits.target.copy()

y[digits.target==9]=1   #转换成极度偏斜的二分类任务
y[digits.target!=9]=0

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

print("==============sklearn中的混淆矩阵,精准率,召回率,F1 Score=======================")
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)
print("分类准确度: ",log_reg.score(X_test,y_test))
y_log_predict=log_reg.predict(X_test)

print("混淆矩阵: \n",confusion_matrix(y_test, y_log_predict))
print("精准率: ",precision_score(y_test, y_log_predict))
print("召回率: ",recall_score(y_test, y_log_predict))
print("F1 Score: ",f1_score(y_test, y_log_predict))

print("=====我们可以适当调整决策边界直线的乘积(thresholds)来调整精准率和召回率的大小，来适应不同的实际情况(类似股票预测和医疗预测)=====")
decision_scores = log_reg.decision_function(X_test)
'''
sklearn没有封装，所以只能手工实现
逻辑回归默认的decision_scores是0
'''
y_predict_2=np.array(decision_scores>=-5,dtype='int')

print("混淆矩阵: \n",confusion_matrix(y_test, y_predict_2))
print("精准率: ",precision_score(y_test, y_predict_2))
print("召回率: ",recall_score(y_test, y_predict_2))
print("F1 Score: ",f1_score(y_test, y_predict_2))

print("===============绘制精准率-召回率曲线=========================")
'''sklearn封装的求precisions, recalls, thresholds'''

'''绘制precisions,recalls,thresholds(y轴)曲线'''
precisions, recalls, thresholds=precision_recall_curve(y_test,decision_scores)
plt.plot(thresholds, precisions[:-1])
plt.plot(thresholds, recalls[:-1])
plt.show()

'''绘制precisions,recalls曲线'''
plt.plot(precisions, recalls)
plt.show()

print("===============绘制ROC曲线=========================")
'''sklearn封装的求fprs, tprs, thresholds'''
fprs, tprs, thresholds = roc_curve(y_test, decision_scores)
plt.plot(fprs, tprs)
plt.show()

'''
ROC曲线下方的面积越大说明我们的分类算法越好
'''
print("ROC AUC(ROC面积): ",roc_auc_score(y_test, decision_scores))

print("================多分类任务的混淆矩阵=======================")
digits=datasets.load_digits()
X=digits.data
y=digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=666)

log_reg=LogisticRegression()    #默认使用ovr的方式解决多分类任务
log_reg.fit(X_train,y_train)
print("分类准确度: ",log_reg.score(X_test,y_test))

y_perdict=log_reg.predict(X_test)

'''多分类任务时要设置average(不能用默认的binary)，有micro，macro，weighted，samples等选项，具体用哪个百度'''
print("精准率: ",precision_score(y_test, y_perdict,average='macro'))
print("召回率: ",recall_score(y_test, y_perdict,average='macro'))
print("F1 Score: ",f1_score(y_test, y_perdict,average='macro'))

print("================绘制多分类任务混淆矩阵=======================")
cfm=confusion_matrix(y_test, y_perdict)
print("混淆矩阵: \n",cfm)

'''把混淆矩阵变化成预测错误的概率矩阵'''
row_sums=np.sum(cfm,axis=1)         #求每一行的和
err_metrix=cfm/row_sums             #每一行的数据除以每一行的和求出每个数字的平均值
np.fill_diagonal(err_metrix,0)      #对角线设置成0

'''plt绘制一个矩阵,cmap将矩阵中的每一个数和一个颜色映射起来'''
'''图像越亮的地方代表出错越多'''
plt.matshow(err_metrix,cmap=plt.cm.gray)   #plt.cm.gray:灰度值
plt.show()
