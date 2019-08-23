import matplotlib.pyplot as plt
import random
from collections import Counter

'''
散点图(二维)：
    两个数值变量    
'''
random.seed(666)
x=[random.randint(0,100) for _ in range(100)]
y=[random.randint(0,100) for _ in range(100)]
plt.scatter(x,y)
plt.show()

'''
折线图(一维)：
    和时间有关的
'''
x=[random.randint(0,100) for _ in range(100)]
plt.plot([i for i in range(100)],x)
plt.show()

'''
条形图(一维)：
    分类变量
'''
data=[3,3,4,1,5,4,2,1,5,4,4,4,5,3,2,1,4,5,5]
counter=Counter(data)
x=[point[0] for point in counter.most_common()]
y=[point[1] for point in counter.most_common()]
plt.bar(x,y)
plt.show()

'''
直方图(一维)：
    数值变量
'''
data=[random.randint(1,100) for _ in range(1000)]

#频数直方图
plt.hist(data,rwidth=0.8,bins=5)
plt.show()
#频率直方图
plt.hist(data,rwidth=0.8,bins=5,density=True)
plt.show()

'''
箱线图(一维)：
    数值变量
'''
data=[random.randint(1,100) for _ in range(1000)]
data.append(200)        #极端值
data.append(-200)
plt.boxplot(data)
plt.show()

#并排箱线图：一个数值变量一个分类变量
data1=[random.randint(66,166) for _ in range(1000)]
data2=[random.randint(60,120) for _ in range(1000)]
plt.boxplot([data1,data2])
plt.show()