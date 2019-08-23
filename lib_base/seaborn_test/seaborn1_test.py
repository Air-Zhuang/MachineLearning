import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
lmplot：         鸢尾花
distplot：       直方图+密度图+分布图
kdeplot：        密度图
rugplot：        分布图
heatmap：        热力图
barplot：        柱状图
jointplot：      散点图(点集中在斜率为45度的对角线上，说明关系强)
pairplot:        多图
countplot:       数量图(Series)
'''

iris=pd.read_csv('../file/iris.csv')
print(iris.head())
print()

'''使用matplotlib画图'''
color_map=dict(zip(iris.Name.unique(),['blue','green','red']))
print(color_map)                        #{'Iris-setosa':'blue','Iris-versicolor':'green','Iris-virginica':'red'}
print()

for species,group in iris.groupby('Name'):
    plt.scatter(group['PetalLength'],group['SepalLength'],color=color_map[species],alpha=0.3,edgecolors=None,label=species)
plt.legend(frameon=True,title='Name')
plt.xlabel('PetalLength')
plt.ylabel('SepalLength')
plt.show()

'''使用seaborn画图'''
sns.lmplot('PetalLength','SepalLength',iris,hue='Name',fit_reg=False)       #按照'Name'分类,fit_reg=False自动画一条趋势线
plt.show()
