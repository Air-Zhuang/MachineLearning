import numpy as np
import matplotlib.pyplot as plt

'''
批量梯度上升法求PCA第一主成分(手工实现)
'''

X=np.empty((100,2))
X[:,0]=np.random.uniform(0.,100.,size=100)
X[:,1]=0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

# plt.scatter(X[:,0],X[:,1])
# plt.show()

def demean(X):
    '''
        将样例的均值归为0：
            X中的每个值减对应列向量的均值
    '''
    return X-np.mean(X,axis=0)


X_demean=demean(X)                  #样例的均值归为0

# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()

print("=================批量梯度上升法使PCA效用函数最大========================")
def f(w,X):
    '''PCA效用函数'''
    return np.sum((X.dot(w)**2)) / len(X)

def df_math(w,X):
    '''批量梯度上升法'''
    return X.T.dot(X.dot(w))*2. / len(X)


def direction(w):
    return w / np.linalg.norm(w)
def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    '''求w过程'''
    w = direction(initial_w)
    cur_iter = 0

    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)                                    #将w转换为单位向量
        if (abs(f(w, X) - f(last_w, X)) < epsilon):         #增加的值有没有超过限度
            break
        cur_iter += 1
    return w


initial_w=np.random.random(X.shape[1])                      #初始w向量不能为0
print("initial_w: ",initial_w)
eta=0.001
'''不能对数据做归一化处理!!!'''

final_w=gradient_ascent(df_math,X_demean,initial_w,eta)     #求出的w叫做第一主成分
print("final_w: ",final_w)

plt.scatter(X_demean[:,0],X_demean[:,1])
plt.plot([0,final_w[0]*30],[0,final_w[1]*30],color='r')
plt.show()