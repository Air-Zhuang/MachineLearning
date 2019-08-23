import numpy as np
import pandas as pd
from pandas import Series,DataFrame

d={
    "Dec 2018":["1","2","3","4","5","6","7","8","9","10"],
    "Dec 2017":["1","2","4","3","7","5","6","9","-","12"],
    "Change":["NaN","NaN","change","change","change","change","change","change","change","change"],
    "Programming Language":["Java","C","Python","C++","Visual Basic .NET","C#","JavaScript","PHP","SQL","Objective-C"],
    "Ratings":["15.932%","14.282%","8.376%","7.562%","7.127%","3.455%","3.063%","2.442%","2.184%","1.477%"],
    "Change.1":["+2.66%","+4.12%","+4.60%","+2.84%","+4.66%","+0.63%","+0.59%","+0.85%","+2.18%","-0.02%"]
}
df=pd.DataFrame(data=d)                                 #根据字典创建
print(df)

print("==================查看=================================")
print(df.shape)                                         #返回DataFrame大小
print(df.columns)                                       #列名
print(type(df.columns))
# print(df.head())                                      #返回前5行
# print(df.tail(6))                                     #返回后6行，默认5行
print("==================转换成列表=================================")
print (df.values)
print("==================聚合操作=================================")
print(df.sum())
print("==================筛选=================================")
print(df['Programming Language'][df.Change == 'change'])        #筛选符合Change列等于'change'的Programming Language列的值
print("==================取值=================================")
# print(df['Change'])                                   #取单列则为Series
# print(df[['Change','Ratings']])                       #取多列则为DataFrame

df_new=DataFrame(df,columns=["Programming Language","Dec 2017","not exist"])            #根据某几列生成新的DataFrame,不存在的列自动赋值NaN
print(df_new)
print()

print(df.iloc[5:9,0:2])                                 #DataFrame切片，返回5-9行，2-3列数据
print()
print(df.iloc[1:9:2,0:2])                               #也可以有步长
print()
print(df.loc['5':'6','Change':'Ratings'])               #DataFrame切片,索引为行和列名
print()

print("==================赋值=================================")
# df_new["not exist"]=range(0,10)                                   #python类型赋值
# df_new["not exist"]=np.arange(0,10)                               #numpy类型赋值
# df_new["not exist"]=pd.Series(np.arange(0,10))                    #Series类型赋值

print("==================修改=================================")
df_new["not exist"]=pd.Series([100,200],index=[1,2])                #只修该第2,3行的数据
print(df_new)

print("==================reindex(修改index)=================================")
df1=DataFrame(np.random.rand(25).reshape(5,-1),index=['A','B','D','E','F'],columns=['c1','c2','c3','c4','c5'])
print(df1)
print()

df1_1=df1.reindex(index=['A','B','c','D','E','F'],columns=['c1','c2','c3','c4','c5','c6'])
print(df1_1)                                                        #没有内容自动赋值NaN
print()

df1_2=df1.reindex(index=['A','B'])                                  #使用reindex实现的裁剪操作
print(df1_2)

print("==================删除行列=================================")
print(df1.drop('A',axis=0))                                         #删除行
print()
print(df1.drop('c1',axis=1))                                        #删除列
print()

print("==================NaN=================================")
# df1_1.isnull()                                                    #生成bool类型DataFrame

'''删除NaN'''
print(df1_1.dropna())                                               #将有NaN值的行和列全部删除
print()
# print(df1_1.dropna(axis=0))                                       #将有NaN值的行全部删除
# print(df1_1.dropna(axis=1))                                       #将有NaN值的列全部删除
print(df1_1.dropna(axis=0,how='all'))                               #只有值全部为NaN才会删除这一行,默认为any
print()
print(df1_1.dropna(axis=0,thresh=2))                                #NaN值得数量大于2才会删除这一行
print()

'''赋值NaN'''
# print(df1_1.fillna(value=1))                                      #将所有的NaN值赋值为1
print(df1_1.fillna(value={0:0,1:1,2:2,3:3}))                        #第一行的NaN赋值为1,第二行的NaN赋值为2......

print("================多级Index==============================")
df2=DataFrame(np.arange(16).reshape(4,-1),index=[['a','a','b','b'],[1,2,1,2]],columns=[['BJ','BJ','SH','GZ'],[8,9,8,9]])
print(df2)                                                          #创建一个多级Index的DataFrame

'''访问'''
print(df2['BJ'])                                                    #访问一级columns是BJ的
print(df2['BJ'][8])                                                 #访问一级columns是BJ,二级columns是8的

print("================Mapping(类似map操作)==============================")
df3=DataFrame({'城市':['北京','上海','广州'],'人口':[1000,2000,3000]},index=['A','B','C'])
print(df3)

# df3['GDP']=Series([1000,2000,1500])                               #通常添加一列,这种方法必须index为0,1,2...

gdp_map={"北京":1000,"上海":2000,"广州":1500}
df3['GDP']=df3['城市'].map(gdp_map)                                 #通过map添加一列
print(df3)

print("================重命名==============================")
df4=DataFrame(np.arange(9).reshape(3,3),index=['BJ','SH','GZ'],columns=['A','B','C'])
print(df4)
print()

# df4.index=Series(['bj','sh','gz'])                                #传统方法修改index
# df4.index.map(str.upper)                                          #通过map方法将index全部变成大写

print(df4.rename(index=str.lower,columns=str.lower))                #使用rename将列名和index转换成小写
print()
print(df4.rename(index={'BJ':'beijing'},columns={'A':'a'}))         #通过映射方式重命名

print("================merge连接(类似Mysql连接)==============================")
df5=DataFrame({'key':['X','Y','Z','X'],'data_set_1':[1,2,3,4]})
print(df5)
print()
df6=DataFrame({'key':['X','Y','C'],'data_set_2':[5,6,7]})
print(df6)
print()

print(pd.merge(df5,df6))                    #合并,列中至少有相同的数据,有相同的值默认遵循内连接

'''
merge参数：
on=''         根据哪一列merge
how='inner'   默认内连接inner,left,right,outer
'''


