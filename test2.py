import pymysql
import numpy as np
import math
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scipy.stats import norm,t,chi2
from mpl_toolkits.mplot3d import Axes3D


data_set2=[[0, 0.5,3.31], [50, 1.0,3.12], [100, 2.0,2.94], [150, 3.0,2.75], [200, 4.0,2.63], [250, 5.0,2.55], [300, 6.0,2.55],
   [350, 7.0,2.46], [400, 8.0,2.48], [450, 9.0,2.51], [500, 10.0,2.5], [600, 20.0,2.31], [700, 30.0,2.19], [800, 35.0,2.36],
   [900, 40.0,2.33], [1000, 45.0,2.2], [1100, 50.0,2.27], [1200, 55.0,2.08], [1300, 60.0,2.29], [1400, 65.0,2.08], [1500, 70.0,2.39],
   [1600, 75.0,2.13], [1700, 80.0,2.02], [1800, 85.0,2.15], [1900, 90.0,1.98], [2000, 95.0,2.07]]

raw_data_x=[[i[1],i[2]] for i in data_set2]
raw_data_y=[i[0] for i in data_set2]


# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter([i[0] for i in data_set2], [i[1] for i in data_set2], [i[2] for i in data_set2], c='r', label='顺序点')
ax.legend(loc='best')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()



X=np.array(raw_data_x)
Y=np.array(raw_data_y)

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=666)
reg=LinearRegression()
reg.fit(X_train,y_train)
print("系数:",reg.coef_)
print("截距:",reg.intercept_)
print("score:",reg.score(X_test,y_test))
