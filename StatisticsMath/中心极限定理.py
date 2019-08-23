import random
import matplotlib.pyplot as plt
from playStats.descriptive_stats import mean

'''
证明中心极限定理:
    中心极限定理说明，在适当的条件下，相互独立的随机变量之和经过适当标准化后(比如均值)，
    其分布近似于正态分布；注意，不要求变量本身服从正态分布
'''

def sample(num_of_samples,sample_sz):
    '''
    返回样本数为10000,样本容量为40的满足均匀分布的样本均值列表
    '''
    data=[]
    for _ in range(num_of_samples):
        data.append(mean([random.uniform(0.0,1.0) for _ in range(sample_sz)]))
    return data

if __name__ == '__main__':
    data=sample(10000,40)
    plt.hist(data,bins='auto',rwidth=0.8)
    plt.axvline(x=mean(data),c='red')      #画一条直线，值为样本均值的均值
    plt.show()