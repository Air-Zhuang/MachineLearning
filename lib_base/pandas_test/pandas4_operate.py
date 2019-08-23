import numpy as np
import pandas as pd
from pandas import Series,DataFrame

print("================加法(NaN+1=NaN)==============================")
s1=Series([1,2,3],index=['A','B','C'])                  #Series相加
s2=Series([4,5,6,7],index=['B','C','D','E'])
print(s1+s2)
print()

df1=DataFrame(np.arange(4).reshape(2,2),index=['A','B'],columns=['北京','上海'])        #DataFrame相加
df2=DataFrame(np.arange(9).reshape(3,3),index=['A','B','C'],columns=['北京','上海','广州'])
print(df1+df2)

print("================聚合函数(NaN+1=1)==============================")
df3=DataFrame([[1,2,3],[4,5,np.nan],[7,8,9]],index=['A','B','C'],columns=['c1','c2','c3'])
print(df3)
print()
print(df3.sum())                #每列求和
print(df3.sum(axis=1))          #每行求和

print(df3.min())                #每列最小值
print(df3.min(axis=1))          #每行最小值

s8=Series([1,2,3,4,5])
print(s8.cumsum())              #相当于reduce

print("================describe(详情)==============================")
print(df3.describe())

print("================排序==============================")
'''Series'''
s1=Series(np.random.randn(10))

s2=s1.sort_values()                             #通过值排序
print(s2)
# s2=s1.sort_values(ascending=False)             #通过值降序排序
s3=s2.sort_index()                                  #通过index排序,DataFrame也可以

'''DataFrame'''
df1=DataFrame(np.random.randn(12).reshape(3,4),columns=['A','B','C','D'])
print(df1)

print(df1.sort_values('A'))                     #根据第一列排序

print("================combine(填充)==============================")
'''Series'''
s6=Series([2,np.nan,4,np.nan],index=['A','B','C','D'])
print(s6)
print()
s7=Series([1,2,3,4],index=['A','B','C','D'])
print(s7)
print()

print(s6.combine_first(s7))                     #用s7来填充s6
print()
'''DataFrame'''
df4=DataFrame({'X':[1,np.nan,3,np.nan],'Y':[5,np.nan,7,np.nan],'Z':[9,np.nan,11,np.nan]})
df5=DataFrame({'Z':[np.nan,10,np.nan,12],'A':[1,2,3,4]})
print(df4)
print(df5)
print()

print(df4.combine_first(df5))                   #用df5来填充df4

print("================concat(拼接)(即将不支持这种操作)==============================")
'''Series'''
s4=Series([1,2,3],index=['X','Y','Z'])
s5=Series([4,5],index=['A','B'])

# print(pd.concat([s4,s5]))             #上下结合生成新Series(即将不支持这种操作)
# print(pd.concat([s4,s5],axis=1))      #左右结合生成新DataFrame(即将不支持这种操作)
'''DataFrame'''
df2=DataFrame(np.random.randn(4,3),columns=['X','Y','Z'])
df3=DataFrame(np.random.randn(3,3),columns=['X','Y','A'])

# print(pd.concat([df2,df3]))             #默认上下组合,没有的值填充NaN(即将不支持这种操作)
