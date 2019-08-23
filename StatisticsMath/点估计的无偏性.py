import random
from playStats.descriptive_stats import mean,variance
import matplotlib.pyplot as plt

"""
证明点估计的无偏性(样本方差用n-1而不是n)
"""

def variance_bias(data):
    """有偏方差(因为除以n)"""
    n=len(data)
    if n<=1:
        return None
    mean_value=mean(data)
    return sum((e-mean_value)**2 for e in data)/n

def sample(num_of_samples,sample_sz,var):
    '''
    返回样本数为num_of_samples,样本容量为sample_sz的方差列表
    '''
    data=[]
    for _ in range(num_of_samples):
        data.append(var([random.uniform(0.0,1.0) for _ in range(sample_sz)]))
    return data

if __name__ == '__main__':
    data1=sample(1000,40,variance_bias)     #有偏方差的情况
    plt.hist(data1,bins="auto",rwidth=0.8)
    plt.axvline(x=mean(data1),c='black')    #基于有偏方差计算出来的均值
    plt.axvline(x=1/12,c='red')             #对于均匀分布来讲(random.uniform),它总体方差的计算公式为(b-a)^2/12
    print("bias: ",mean(data1),1/12)
    plt.show()

    data2 = sample(1000, 40, variance)      #无偏方差的情况
    plt.hist(data2, bins="auto", rwidth=0.8)
    plt.axvline(x=mean(data2), c='black')   #基于无偏方差计算出来的均值
    plt.axvline(x=1 / 12, c='red')          #对于均匀分布来讲(random.uniform),它总体方差的计算公式为(b-a)^2/12
    print("unbias: ", mean(data2), 1 / 12)
    plt.show()