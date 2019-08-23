import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

df_e=pd.read_excel('../file/sales-funnel.xlsx')
print(df_e.columns)
print(df_e.head())

print("==================透视表(维度转换)=================================")
print(pd.pivot_table(df_e,index=['Name']).head())                 #因为聚合方法默认取平均值,所以只处理数字类型列
print()
print(pd.pivot_table(df_e,index=['Name'],aggfunc='sum').head())   #聚合方法改为求和
print()
print(pd.pivot_table(df_e,index=['Manager','Rep']).head())        #生成多级index的透视表
print()
print(pd.pivot_table(df_e,index=['Name'],columns='Status',values='Price',aggfunc='sum').head())      #纵坐标改成Name，横坐标改成Status，统计值为Price
print()

'''
pivot_table参数：
index=None          行index
aggfunc='mean'      聚合方法，默认取平均值
values=None         聚合操作的内容
columns=None        列index
fill_value=None     NaN值填充
'''

print("==================Demo=================================")
df=pd.read_csv('../file/usa_flights.csv')
print(df.columns)
print(df.head())

print("+++++++++++++++++++1.获取延误时间最长top10(sort_values)++++++++++++++++++++++++++++++++")
print(df.sort_values('arr_delay',ascending=False)[:10])         #ascending=False  倒序排列

print("+++++++++++++++++++2.计算延误和没有延误所占比例(value_counts)++++++++++++++++++++++++++++++++")
df['delayed']=df['arr_delay'].apply(lambda x:x>0)
print(df.head())
print()
delay_data=df['delayed'].value_counts()
print(delay_data)
print()
print("延误比：",delay_data[1]/(delay_data[0]+delay_data[1]))

print("+++++++++++++++++++3.每一个航空公司延误的情况++++++++++++++++++++++++++++++++")
delay_group=df.groupby(['unique_carrier','delayed'])
print(delay_group.size())                           #求数量，生成一个Series
print()
df_delay=delay_group.size().unstack()               #将多级index的Series转换成DataFrame
print(df_delay)

df_delay.plot(kind='barh',stacked=True,figsize=[16,6],colormap='winter')        #做图
plt.show()

print("+++++++++++++++++++4.透视表功能++++++++++++++++++++++++++++++++")
flights_by_carrier=df.pivot_table(index='flight_date',columns='unique_carrier',values='flight_num',aggfunc='count')
print(flights_by_carrier.head())