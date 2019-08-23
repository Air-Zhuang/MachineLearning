import numpy as np
from scipy.linalg import svd

print("==============SVD分解=================")
A=np.array([[1,2],[3,4],[5,6]])

U,s,VT=svd(A)
print(U)                        #3*3
print(s)                        #只返回向量，需要自己拼接成3*2的矩阵
print(VT)                       #2*2
print()

Sigma=np.zeros(A.shape)
for i in range(len(s)):
    Sigma[i][i]=s[i]            #s拼接成3*2的矩阵

print(U.dot(Sigma).dot(VT))     #证明分解出的SVD相乘是原矩阵
