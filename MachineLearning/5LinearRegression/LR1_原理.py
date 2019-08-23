import numpy as np
import matplotlib.pyplot as plt

'''
简单线性回归原理(手工实现)
'''

x=np.array([1.,2.,3.,4.,5.])
y=np.array([1.,3.,2.,3.,5.])

x_mean=np.mean(x)                   #x均值
y_mean=np.mean(y)                   #y均值

print("===========求回归方程中的a和b===============")
num=(x-x_mean).dot(y-y_mean)        #求a公式中的分子
d=(x-x_mean).dot(x-x_mean)          #求a公式中的分母

a=num/d                             #a的值
b=y_mean-a * x_mean                 #b的值
print("a: ",a,"b: ",b)

y_hat=a*x+b
print("y_hat:",y_hat)
plt.scatter(x,y)                    #散点图
plt.plot(x,y_hat,color='r')         #使用简单线性回归得到的直线
plt.axis([0,6,0,6])                 #坐标
# plt.show()

x_predict=6
y_predict=a*x_predict+6
print("x=6的预测结果: ",y_predict)

