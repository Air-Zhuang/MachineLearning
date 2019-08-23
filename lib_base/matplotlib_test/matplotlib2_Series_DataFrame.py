import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


print("==============Series绘图=================")
s1=Series(np.random.randn(1000)).cumsum()               #相当于reduce操作
s2=Series(np.random.randn(1000)).cumsum()
print(s1.head(10))

'''一张图绘制多条线'''
s1.plot(label='s1')
s2.plot(label='s2')
plt.legend()
plt.show()

'''子图subplots,一张图绘制多种类型图'''
fig,ax=plt.subplots(2,1)
s1[0:10].plot(ax=ax[0],label='s1',kind='bar')           #柱形图
s2.plot(ax=ax[1],label='s2')
plt.legend()
plt.show()

print("==============DataFrame绘图=================")
df=DataFrame(np.random.randint(1,10,40).reshape(10,-1),columns=['A','B','C','D'])
print(df.head())

df.plot(kind='barh',stacked=True)                       #横向堆叠的柱状图
plt.show()

df.T.plot(kind='barh',stacked=True)                     #行列转置后的柱状图
plt.show()