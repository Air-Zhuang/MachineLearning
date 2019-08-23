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
df1=pd.DataFrame(data=d)                            #根据字典创建
print(df1)

print("==================粘贴板=================================")
# df1.to_clipboard()                                #将DataFrame写入到粘贴板中
# df=pd.read_clipboard()                            #从粘贴板获取
# print(df)

print("==================csv=================================")
# df1.to_csv('df1.csv',index=False)                 #转储csv不写入index
# df2=pd.read_csv('df1.csv',index_col='Date')       #读取csv,以'Date'为index
# print(df2)

print("==================json=================================")
# df_json=df1.to_json()                             #转为json
# print(df_json)
# pd.read_json(df_json)                             #读取json

print("==================html=================================")
# df1.to_html('df1.html')                           #转储html
# pd.read_html('df1.html')                          #读取html

print("==================excel=================================")
# df1.to_excel('df1.xlsx')                          #转储excel
# pd.read_excel('df1.xlsx')                         #读取excel

