import pandas_datareader as pdr
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# start=datetime(2015,9,20)
# alibaba=pdr.get_data_yahoo('BABA',start=start)                            #在线获取https://finance.yahoo.com的股票信息
# alibaba.to_csv('alibaba.csv')
# amazon=pdr.get_data_yahoo('AMZN',start=start)
# amazon.to_csv('amazon.csv')

alibaba=pd.read_csv('../file/BABA.csv',index_col='Date')
amazon=pd.read_csv('../file/AMZN.csv',index_col='Date')
print(alibaba.head())
print()
print(amazon.head())
print()

# print(alibaba.describe())                                                 #内置的基本的数据分析

print("+++++++++++++++++++收盘价图++++++++++++++++++++++++++++++++")
alibaba['Adj Close'].plot(legend=True)                                      #收盘价图,legend=True加上label
amazon['Adj Close'].plot(legend=True)
plt.show()

print("+++++++++++++++++++交易量图++++++++++++++++++++++++++++++++")
alibaba['Volume'].plot(legend=True)                                         #交易量图
amazon['Volume'].plot(legend=True)
plt.show()

print("+++++++++++++++++++每天之内最高价最低价差值变化情况图++++++++++++++++++++++++++++++++")
alibaba['high-low']=alibaba['High']-alibaba['Low']
alibaba['high-low'].plot()
plt.show()

print("+++++++++++++++++++每天收盘价之间变化情况图++++++++++++++++++++++++++++++++")
alibaba['daily-return']=alibaba['Adj Close'].pct_change()                   #生成变化情况Series
alibaba['daily-return'].plot(figsize=(10,4),marker='o')
plt.show()

'''绘制seaborn图'''
sns.distplot(alibaba['daily-return'].dropna(),bins=100,color='purple')
plt.show()