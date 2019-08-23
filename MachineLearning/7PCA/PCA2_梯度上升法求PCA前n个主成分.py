import numpy as np
import matplotlib.pyplot as plt

'''
批量梯度上升法求PCA前n个主成分(手工实现)
'''

X=np.empty((100,2))
X[:,0]=np.random.uniform(0.,100.,size=100)
X[:,1]=0.75*X[:,0]+3.+np.random.normal(0,10.,size=100)

plt.scatter(X[:,0],X[:,1])
plt.show()

def demean(X):
    '''
        将样例的均值归为0：
            X中的每个值减对应列向量的均值
    '''
    return X-np.mean(X,axis=0)


X=demean(X)                  #样例的均值归为0

# plt.scatter(X_demean[:,0],X_demean[:,1])
# plt.show()

print("=================批量梯度上升法使PCA效用函数最大========================")
def f(w,X):
    '''PCA效用函数'''
    return np.sum((X.dot(w)**2)) / len(X)

def df(w,X):
    '''批量梯度上升法'''
    return X.T.dot(X.dot(w))*2. / len(X)


def direction(w):
    return w / np.linalg.norm(w)
def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
    '''求第一主成分的过程'''
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
eta=0.01
'''不能对数据做归一化处理!!!'''

print("==============求第一主成分==========================")
w=first_component(X,initial_w,eta)                          #求出的w叫做第一主成分
print("w: ",w)

X2=X-X.dot(w).reshape(-1,1)*w                               #X2是把X在求得的第一主成分上相应的分量去掉的结果

plt.scatter(X2[:,0],X2[:,1])
plt.show()

print("==============求第二主成分==========================")
w2=first_component(X2,initial_w,eta)
print("w2: ",w2)
print("w和w2是垂直的(点乘结果为0): ",w.dot(w2))


print("==============对上面求前n个主成分的封装==========================")
def first_n_components(n, X, eta=0.01):
    X_pca = X.copy()
    X_pca = demean(X_pca)
    res = []
    for i in range(n):
        initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)
        X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w
    return res

w_list=first_n_components(2,X)
print("w_list: ",w_list)