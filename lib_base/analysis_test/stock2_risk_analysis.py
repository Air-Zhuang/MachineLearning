import pandas_datareader as pdr
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

top_tech_df=pd.read_csv('../file/top5.csv',index_col='Date').sort_index()
print(top_tech_df.head())
print()

print("+++++++++++++++++++风险分析++++++++++++++++++++++++++++++++")
top_tech_dr=top_tech_df.pct_change()                    #通过pct_change生成每日变化情况
print(top_tech_dr.head())
print()

top_tech_df.plot()                          #所有公司的股票走势图
plt.show()

top_tech_df[['AAPL','FB','MSFT']].plot()        #三家公司的股票走势图
plt.show()

sns.jointplot('AMZN','GOOG',top_tech_dr,kind='scatter')         #散点图(点集中在斜率为45度的对角线上，说明两家股票相关性强)
plt.show()

'''jointplot:x,y两个数据的对比'''
sns.jointplot('MSFT','FB',top_tech_dr,kind='scatter')         #相关性不强
plt.show()

sns.jointplot('AMZN','FB',top_tech_dr,kind='scatter')
plt.show()

'''pairplot:生成多张图'''
sns.pairplot(top_tech_dr.dropna())
plt.show()
