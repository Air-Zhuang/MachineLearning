import random
import matplotlib.pyplot as plt
from playStats.descriptive_stats import mean,variance

"""
证明点估计的相合性(样本均值和方差在样本容量逐渐变大的情况下稳定于真值)
"""

if __name__ == '__main__':
    sample_means=[]     #样本均值
    sample_vars=[]      #样本方差
    indices=[]          #样本容量

    for sz in range(20,10001,50):
        indices.append(sz)
        sample=[random.gauss(0.0,1.0) for _ in range(sz)]   #生成样本容量个数满足标准正态分布的样本
        sample_means.append(mean(sample))           #计算样本均值
        sample_vars.append(variance(sample))        #计算样本方差

    plt.plot(indices,sample_means)
    plt.plot(indices,sample_vars)
    plt.show()
