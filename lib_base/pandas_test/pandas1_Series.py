import numpy as np
import pandas as pd
from pandas import Series,DataFrame

print("================创建==============================")
s1=pd.Series([1,2,3,4])                             #通过pandas创建series
print(s1)
print(s1.values)                                    #是numpy类型的数组
print(s1.index)                                     #查看索引
print()

s2=pd.Series(np.arange(4))                          #通过numpy创建series,(只能是一维矩阵)
print(s2)
print()

s3=pd.Series({"A":1,"B":2,"C":3})                   #通过字典创建
print(s3)
print()

s4=pd.Series([1,2,3,4],index=['A','B','C','D'])     #指定index
print(s4)

print("================取值==============================")
print(s4['A'])                                      #取键为A的值
print(s4[s4>2])                                     #取值大于2的

print("================操作==============================")
print(s4.to_dict())                                 #转换成字典

index_1=['A','B','C','D','E']
s5=pd.Series(s4,index=index_1)                      #键不存在的自动赋值NaN
print(s5)
print()

print(pd.isnull(s5))                                #生成bool类型的Series
# print(pd.notnull(s5))
print()

s4.name="demo"                                      #series起名
s4.index.name="demo index"                          #series的index起名
print(s4)

'''删除'''
# s4.drop('A')                                      #通过索引删除行
print(s5.dropna())                                  #将有NaN值的行删除

print("================reindex(修改index)==============================")
s5=pd.Series([1,2,3,4],index=['A','B','C','D'])

# s5.reindex(index=['A','B','C','D','E'])               #重新设置index,索引没有内容自动赋值NaN
ss5=s5.reindex(index=['A','B','C','D','E'],fill_value=10)             #重新设置index,没有内容自动赋值10
print(ss5)
print()

s6=pd.Series(['A','B','C'],index=[1,3,5])
s6_1=s6.reindex(index=range(7))                         #内容没有则赋值NaN
print(s6_1)
print()

s6_2=s6.reindex(index=range(7),method='ffill')          #用这种方式可以连续赋值
print(s6_2)

print("================多级Index==============================")
s7=Series(np.random.randn(6),index=[['1','1','1','2','2','2'],['a','b','c','a','b','c']])
print(s7)
print()

print(type(s7['1']))                    #通过一级index获取一个Series
print(s7['1']['a'])                     #获取第一个Series的某一行值
print(s7[:,'a'])                        #获取所有Series的'a'字段，返回值为一个Series

print("================Series DataFrame互相转换==============================")
# print(DataFrame(s7['1'],s7['2']))     #通过多级Index的Series创建DateFrame

df7=s7.unstack()                        #Series转DataFrame
print(df7)
print()

s8=df7.unstack()                        #DataFrame转Series
print(s8)
print()
s8=df7.T.unstack()                      #DataFrame转Series,行列互换
print(s8)

print("================Replace(值替换)==============================")
s9=Series(np.arange(10))

print(s9.replace(1,np.nan))                    #值为1的替换成NaN
s9.replace({1:np.nan})                  #值为1的替换成NaN
s9.replace([1,2,3],[10,20,30])          #值为1,2,3的对应替换成10,20,30




