import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series,DataFrame

print("==============直方图/密度图/分布图=================")
s1=Series(np.random.randn(1000))

'''
hist直方图,kde密度图,rug分布图,bins分区数
'''
sns.distplot(s1,hist=True,kde=True,rug=True,bins=20)        #distplot可以同时显示直方图,密度图,分布图
plt.show()

'''
shade是否填充,color颜色
'''
sns.kdeplot(s1,shade=True)                                  #kdeplot主要画密度图
plt.show()

sns.rugplot(s1)                                             #rugplot主要画分布图
plt.show()

print("==============热力图=================")
df=sns.load_dataset('flights')                              #在线获取数据
print(df.head())
print()

df=df.pivot(index='month',columns='year',values='passengers')   #生成新的透视表
print(df.head())

'''
annot每个方格显示数据,fmt方格显示数据类型
'''
sns.heatmap(df)                                             #热力图
plt.show()

print("==============柱状图=================")
s=df.sum()                                                  #先将DataFrame转换为Series
print(s.head())                                             #每年乘客的总和

sns.barplot(x=s.index,y=s.values)                           #柱状图
plt.show()
print()

l=[('Fashion', 3694), ('Lifestyle', 3682), ('Travel', 3616), ('Fitness', 1880), ('Beauty', 1854), ('Photography', 1617), ('Fashion & Beauty', 1196), ('Food and Drink', 1152), ('Fashion Blogger', 1033), ('Music', 1029)]
s4=pd.Series(data=[i[1] for i in l],index=[i[0] for i in l])
print(s4.head())
sns.barplot(x=s4.values,y=s4.index, orient='h')             #横向柱状图
plt.show()