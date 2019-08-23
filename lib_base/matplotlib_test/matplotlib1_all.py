import matplotlib.pyplot as plt
import numpy as np
from pandas import Series,DataFrame

x=np.linspace(0,10,100)                             #0-10均分100为数组

cosy=np.cos(x)
siny=np.sin(x)

'''
plot参数：
ax=None         如果绘制subplots子图,此参数指定绘制哪张图
kind='line'     绘制什么图,默认连续图  line连续图 bar柱状图 barh横向柱状图 area填充线型图
stacked=False   图像是否堆叠(柱状图)
grid=False      是否展示网格,默认不展示
rwidth=1        柱状图宽度占比,柱于柱之间空隙(不只是柱状图)
bins=20         分割多少
figsize=(10,4)  图大小
marker='o'      每个取值点加粗显示
'''

print("==============连续图plot=================")
plt.plot(x,siny,label="sin(x)")                     #在一张图上绘制两条曲线,可以指定曲线标签
plt.plot(x,cosy,color="red",linestyle="--",label="cos(x)")         #可以指定颜色,可以指定线条
plt.xlim(-2,12)                                     #可以指定x轴的显示范围
plt.ylim(-1.2,1.2)                                  #可以指定y轴的显示范围
# plt.axis([-2,12,-1.2,1.2])                        #可以同时指定x和y的范围
plt.xlabel("x yield")                               #可以指定标签文本
plt.ylabel("x yield")
plt.legend()                                        #加上这条才会展示曲线上的标签(label)
plt.title("Welcome to the ML World!")               #可以加上标题
plt.show()

print("==============散点图scatter=================")
x=np.random.normal(0,1,5000)
y=np.random.normal(0,1,5000)
plt.scatter(x,y,alpha=0.5,marker="x")               #可以指定点的透明度,可以指定点的形状
plt.show()                                          #绘制一个标准的二维正态分布散点图

print("==============直方图=================")
s=Series(np.random.randn(1000))                 #符合标准正态分布
plt.hist(s,rwidth=0.9,bins=20)
plt.show()

print("==============密度图=================")
s.plot(kind='kde')
plt.show()

print("==============子图subplots(通常用这种)=================")
x=np.linspace(0,10,100)

figure,ax=plt.subplots(2,2)
ax[0][0].plot(x,siny,color="red")
ax[0][1].plot(x,cosy,color="blue")
ax[1][0].plot(x,siny,color="green")
ax[1][1].plot(x,cosy,color="black")
plt.show()

print("==============子图subplot=================")
plt.subplot(2,1,1)                                  #定义一个两行一列的subplot，切换到第一张子图进行绘制
plt.plot(x,siny,color="red")
plt.ylabel("subplot siny")

plt.subplot(2,1,2)
plt.plot(x,cosy,color="blue")
plt.ylabel("subplot cosy")
plt.xlabel("subplot x")

plt.show()
