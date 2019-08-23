import numpy as np
import matplotlib.pyplot as plt

'''
模拟梯度下降法(手工实现)
'''

def dJ(theta):
    return 2*(theta-2.5)            #返回在theta这一点对应的导数


def J(theta):                       #返回theta这一点对应的损失函数
    return (theta-2.5)**2-1


eta=0.01                            #学习率,一般设置成0.01就可以
theta=0.0
theta_history=[theta]
epsilon=1e-8                        #精度
while True:
    gradient=dJ(theta)              #每次求得的导数
    last_theta=theta
    theta=theta-eta*gradient        #向导数的负方向移一步(移动距离=导数*学习率)
    theta_history.append(theta)

    if abs(J(theta)-J(last_theta))<epsilon:     #如果两次损失函数之间的差值小于精度，则可以确定theta
        break
print(theta)                        #x轴位置
print(J(theta))                     #y轴位置
print("多少次移动: ",len(theta_history))

plot_x=np.linspace(-1,6,141)        #在-1到6等截距出141个点
plt.plot(plot_x,J(plot_x))
plt.plot(np.array(theta_history),J(np.array(theta_history)),color='r',marker="+")
plt.show()

print("================手工封装==========================")
def gradient_descent(initial_theta, eta, epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)

    while True:
        gradient = dJ(theta)            #每次求得的导数
        last_theta = theta
        theta = theta - eta * gradient  #向导数的负方向移一步(移动距离=导数*学习率)
        theta_history.append(theta)

        if abs(J(theta) - J(last_theta)) < epsilon:   #如果两次损失函数之间的差值小于精度，则可以确定theta
            break

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color="r", marker='+')
    plt.show()

eta=0.8
theta_history=[]
gradient_descent(0,eta)
plot_theta_history()