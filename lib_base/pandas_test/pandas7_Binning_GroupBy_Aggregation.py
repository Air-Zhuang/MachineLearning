import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import random

def gen_rand_str():
    s_list='QWERTYUIOPASDFGHJKLZXCVBNM'
    s=''
    for i in range(3):
        s+=random.choice(s_list)
    return s

print("==================数据分箱技术Binning(多用于求分布情况)=================================")
print("+++++++++++++++++++对numpy类型数据分箱++++++++++++++++++++++++++++++++")
score_list=np.random.randint(25,100,size=100)

bins=[0,59,70,80,100]

score_cat=pd.cut(score_list,bins=bins)                      #numpy类型分箱,返回一个Categories类型对象
print(score_cat)
print()
print(type(score_cat))                                      #Categories类型实际是Series子类
print()

print(pd.value_counts(score_cat))                           #Categories类型转换成Series类型(统计Categories分布情况)
print()

print("+++++++++++++++++++对DataFrame类型数据分箱++++++++++++++++++++++++++++++++")
df=DataFrame()
df['score']=score_list
df['student']=[gen_rand_str() for i in range(100)]
print(df.head())
print()

print(pd.cut(df['score'],bins=bins).head())                 #DataFrame类型分箱,返回一个Categories类型对象
print()

df_cate=pd.cut(df['score'],bins=bins,labels=['0-59','59-70','70-80','80-100'])         #每一种类型加一个标签
print(df_cate.head())
print()
print(type(df_cate))                                        #Categories类型实际是Series子类
print()
df['Categories']=df_cate
print(df.head())
print()
print(pd.value_counts(df_cate))                             #Categories类型转换成Series类型

print("==================数据分组技术GroupBy=================================")
df1=pd.read_csv('../file/city_weather.csv')
print(df1.head(8))
print()

print("+++++++++++++++++++单列分组++++++++++++++++++++++++++++++++")
g=df1.groupby(df1['city'])                  #生成一个DataFrameGroupBy类型数据

print(g.get_group('BJ'))                    #生成BJ分组的新DataFrame
print()
print(g.max())                              #对Group类型聚合生成DataFrame类型(聚合操作)
print()
print(g.get_group('BJ').max())              #对DataFrame类型聚合生成Series类型
print("|||||||||")
print(list(g))                              #转换为python的list
print()
print(dict(list(g)))                        #转换为python的dict

print("+++++++++++++++++++多列分组++++++++++++++++++++++++++++++++")
g_new=df1.groupby(['city','wind'])
print(g_new.get_group(('BJ',2)))            #生成BJ,wind为2分组的新DataFrame
print()

for (name_1,name_2),group in g_new:         #所有分组信息
    print(name_1,name_2)
    print(group)

print("==================数据聚合技术Aggregation=================================")
print(g.mean())                             #对Group类型求平均值

def foo(attr):
    return attr.max()-attr.min()


print(g.agg(foo))                           #自定义聚合操作：每组最大值和最小值的差


