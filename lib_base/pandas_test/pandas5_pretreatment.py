import numpy as np
import pandas as pd
from pandas import Series,DataFrame

df=pd.read_csv('../file/apply_demo.csv')
print("DataFrame大小：",df.size)
print(df.head())

print("==================apply批量修改(预处理)=================================")
s1=Series(['a']*7978)
df['A']=s1                                                          #添加一列
print(df.head())
print()

df['A']=df['A'].apply(str.upper)                                    #使用apply将A这一列全部转换成大写

print("+++++++++++++++++++预处理：将data字段提取APPL++++++++++++++++++++++++++++++++")
df1=pd.read_csv('../file/apply_demo.csv')
def fo(line):
    items=line.strip().split(' ')
    return items[1]


df1['data'] = df1['data'].apply(fo)                                 #使用apply做整列预处理
print(df1.head())

print("+++++++++++++++++++预处理：将data字段分成三列,分别起名++++++++++++++++++++++++++++++++")
def foo(line):
    items=line.strip().split(' ')
    return Series([items[1],items[3],items[5]])


df_tmp = df['data'].apply(foo)                                      #将data列分成三列赋值给一个临时表
print(df_tmp.head())
print()

df_tmp=df_tmp.rename(columns={0:"Symbol",1:"Seqno",2:"Price"})      #临时表重命名
print(df_tmp.head())
print()

df_new=df.combine_first(df_tmp)                                     #用新表填充旧表
df_new=df_new.drop(['data','A'],axis=1)                             #删除表中重复的 data 字段

print(df_new.head())
print("+++++++++++++++++++预处理：另一种方式++++++++++++++++++++++++++++++++")
def fo1(line):
    items=line.strip().split(' ')
    return items[1]
def fo2(line):
    items=line.strip().split(' ')
    return items[3]
def fo3(line):
    items=line.strip().split(' ')
    return items[5]

df2=pd.read_csv('../file/apply_demo.csv')
df2["A"]=df2["B"]=df2["C"]=np.nan
print(df2.head())
df2["A"]=df2["data"].apply(fo1)
df2["B"]=df2["data"].apply(fo2)
df2["C"]=df2["data"].apply(fo3)
df2=df2.drop(['data'],axis=1)
print(df2.head())

print("==================数据清洗=================================")
print("+++++++++++++++++++去掉Seqno重复的字段++++++++++++++++++++++++++++++++")
print(len(df_new['Seqno'].unique()))                                #查看Seqno列非重复数据的数量
# print(df_new['Seqno'].duplicated())                               #根据是否重复返回一个bool类型Series

print(len(df_new.drop_duplicates()))                              #执行去重(只去重完全重复内容的行)
print(len(df_new.drop_duplicates('Seqno')))                       #执行去重(根据某一字段去重，不管其他字段是否重复)
'''
drop_duplicates参数：
keep='first'      如果重复保留哪条数据，默认第一条first   first/last
'''