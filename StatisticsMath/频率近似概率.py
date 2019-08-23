import random
import matplotlib.pyplot as plt

"""
模拟抛硬币验证小数定律和大数定律
证明为什么可以用频率近似概率
"""

def toss():
    """模拟抛硬币"""
    return random.randint(0,1)

if __name__ == '__main__':
    indices=[]      #存储每波抛了多少次硬币
    freq=[]         #存储每波正面朝上的频率

    for toss_num in range(10,10001,10):
        heads=0     #正面朝上的次数
        for _ in range(toss_num):
            if toss()==0:
                heads+=1
        freq.append(heads/toss_num)
        indices.append(toss_num)

    plt.plot(indices,freq)
    plt.show()