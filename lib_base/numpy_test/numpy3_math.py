'''
支持计算：
x为矩阵

+  -  *  /  //  %  **  1/x(取倒数)
abs()                           绝对值
sin() cos() tan()               三角函数
exp(x)                          e的x次方
power(3,x)                      3的x次方
log(x) log2() log10()           以e,2,10为底
'''
import numpy as np
from numpy.linalg import eig,inv

print("==============向量之间运算=================")
vec=np.array([1,2,3])
vec2=np.array([4,5,6])
print("向量相加:",vec+vec2)
print("向量相减:",vec-vec2)
print("向量数量乘:",2*vec)
print("np中向量相乘:",vec*vec2)
print("向量点乘:",vec.dot(vec2))
print("向量的模:",np.linalg.norm(vec))
print("计算单位向量:",vec / np.linalg.norm(vec))
print("列向量乘行向量:\n",np.array([[1],[2],[3]]).dot(np.array([[1,2]])))

print("==============矩阵之间运算=================")
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
vec3=np.array([10,100])
print("矩阵的转置:\n",A.T)
print("矩阵相加:\n",A+B)
print("矩阵相减:\n",A-B)
print("矩阵数量乘:\n",10*A)
print("np中矩阵相乘:\n",A*B)
print("矩阵相乘:\n",A.dot(B))
print()
print("np中矩阵和向量相加:\n",A+vec3)
print("np中矩阵和数字相加:\n",A+1)
print("矩阵和向量相乘:\n",A.dot(vec3))

'''
Tensorflow:
列向量(15*1)*行向量(1*10)=矩阵(15*10)
矩阵(15*10)+列向量(15*1)/行向量(1*10)=矩阵(15*10)

矩阵(15*10)*列向量(10*1)=列向量(15*1)
列向量(15*1)+列向量(15*1)=列向量(15*1)
'''

print("==============单位矩阵=================")
print(np.eye(4))
print("==============逆矩阵(矩阵必须为方矩阵)=================")
A=np.array([[0,1],[2,3]])
a=np.linalg.inv(A)                              #求A的逆矩阵(逆矩阵与原矩阵相乘结果为单位矩阵)
print("逆矩阵:\n",a)
print(a.dot(A))                                 #逆矩阵与原矩阵相乘结果为单位矩阵
print(A.dot(a))                                 #逆矩阵与原矩阵相乘结果为单位矩阵
print("==============伪逆矩阵(对不为方阵的矩阵求逆矩阵)=================")
C=np.arange(16).reshape(2,8)
c=np.linalg.pinv(C)
print("伪逆矩阵:\n",c)
print(C.dot(c))                                 #原矩阵与逆矩阵(这里必须是原乘逆)相乘结果为2*2的伪单位矩阵，近似为单位矩阵

print("==============求解特征值特征向量=================")
A1=np.array([[4,-2],[1,1]])
eigenvalues1,eigenvectors1=eig(A1)
print("特征值(2个):\n",eigenvalues1)
print("特征向量(2个,一列对应一组,已经做好归一化处理):\n",eigenvectors1)
print()

A2=np.array([[0,-1],[1,0]])
eigenvalues2,eigenvectors2=eig(A2)
print("特征值是复数(2个):\n",eigenvalues2)
print("特征向量是复数(2个,一列对应一组,已经做好归一化处理):\n",eigenvectors2)

print("==============求解矩阵对角化=================")
def diagonalize(A):
    '''A=PDP-1'''
    assert A.shape[0]==A.shape[1]       #需要是一个方阵
    eigenvalues, eigenvectors = eig(A)
    P=eigenvectors
    D=np.diag(eigenvalues)              #通过传入的值生成一个对角矩阵
    Pinv=inv(P)                         #P的逆
    return P,D,Pinv

A3=np.array([[1,2,3,3],[0,2,1,2],[3,1,3,1],[3,5,6,9]])
P1,D1,Pinv1=diagonalize(A3)
e=P1.dot(D1).dot(Pinv1)                 #证明相乘等于A3
# e=np.around(e, decimals=2, out=None)    #保留两位有效数字
print(e)

print("==============聚合操作=================")
L=np.arange(16).reshape(4,-1)
print(L)
print(np.min(L),"最小值")                          #最小值
print(np.max(L),"最大值")                          #最大值
print(np.sum(L),"求和")                            #求和(比普通sum效率更高)
print(np.sum(L,axis=0),"每一列求和")               #每一列求和(axis=0表示沿着第一个维度，行，就是沿着行求和，结果就是每一列的和)
print(np.sum(L,axis=1),"每一行求和")               #每一行求和
print(np.prod(L+1),"乘积")                         #矩阵所有元素的乘积
print(np.mean(L),"平均值")                         #平均值
print(np.median(L+1),"中位数")                     #中位数
print(np.percentile(L,q=50),"百分位")              #百分位，q=50相当于中位数值，q=100相当于最大值
for percent in [0,25,50,75,100]:                   #通常通过5个百分位点可以大致看出一个样本的分布情况
    print(np.percentile(L,q=percent),end=" ")
print()
print(np.var(L),"方差")                            #方差
print(np.std(L),"标准差")                          #标准差
