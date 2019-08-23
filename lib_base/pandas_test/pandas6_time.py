import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

from datetime import datetime

print("==================时间序列创建=================================")
'''通过python的datetime库生成时间对象index'''
data_list=[datetime(2016,9,1),datetime(2016,9,10),datetime(2017,9,1),datetime(2017,9,20),datetime(2017,10,1)]
print(data_list)                                                    #列表中都是时间对象

s1=Series(np.random.randn(5),index=data_list)                       #通过时间对象作为index创建Series
print(s1)
print(s1.index)                                                     #index的类型为DatetimeIndex
print()

'''通过pandas的data_range方法生成时间对象index'''
data_list_new=pd.date_range('2016-01-01',periods=100,freq='5H')
'''
date_range参数：
start=None          开始时间
end=None            结束时间
periods=None        取多少次
freq='D'            步长类型,默认'天' H(5H),D,W(W-MON从周一开始数),M,Y
'''
s2=Series(np.random.rand(100),index=data_list_new)                  #通过时间对象作为index创建Series
print(s2.head())

print("==================时间序列取值=================================")
print(s1[1])                                                        #通过索引位置取值
print(s1[datetime(2016,9,10)])                                      #通过时间对象取值
print(s1['2016-09-10'])                                             #通过字符串类型时间取值
print(s1['20160910'])                                               #通过字符串类型时间取值
print(s1['2016-09'])                                                #返回九月所有数据作为Series
print(s1['2016'])                                                   #返回2016年所有数据作为Series

print("==================时间序列采样和填充=================================")
t_range=pd.date_range('2016-01-01','2016-12-31')
s3=Series(np.random.randn(len(t_range)),index=t_range)
print(s3.head())
print()

print('''采样数据''')
s3_month=s3.resample('M').mean()                                    #采样每个月的<平均值>(pandas提供的时间采样方法)
print(s3_month)
print()
print('''填充数据''')
s3_hour=s3.resample('H').ffill()                                    #从之后的值填充
s3.resample('H').bfill()                                            #从之前的值填充
print(s3_hour.head())
print()
print('''每日根据前日变化情况''')
print(s3.pct_change().head())
print("==================画图=================================")
t_range=pd.date_range('2016-01-01','2016-12-31',freq='H')
print('长度:',len(t_range))
stock_df=DataFrame(index=t_range)                                   #构造时间序列为index的空DataFrame
stock_df['BABA']=np.random.randint(80,160,size=8761)                #填充股票数据
stock_df['TENCENT']=np.random.randint(30,50,size=8761)
print(stock_df.head())
print()

weekly_df=DataFrame()
weekly_df['BABA']=stock_df['BABA'].resample('W').mean()             #周采样
weekly_df['TENCENT']=stock_df['TENCENT'].resample('W').mean()       #周采样
print(weekly_df.head())

weekly_df.plot()
plt.show()


