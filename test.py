import pymysql
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def calculate_fans_num(s):
    if s:
        if s.endswith('k'):
            return int(float(s[:-1])*1000)
        if s.endswith('m'):
            return int(float(s[:-1])*1000000)
        if s.endswith(".0"):
            return int(s[:-2])
        return int(s)

def handle_price(s):
    if "," in str(s):
        return int(s.replace(",", ""))
    return int(s)

def handle_rate(s):
    if "%" in str(s):
        return float(str(s).replace("%", ""))
    return float(s)

def handle_gender(s):
    if s=="Female":
        return 1
    else:
        return 0


data_set=[]
db = pymysql.connect(host="127.0.0.1", user="root", password="123456",db="kols_original", port=3306)
cur = db.cursor()
try:
    sql='select fans_num,engagement_rate,price,gender from bak_influence_table where price is not null and engagement_rate is not null and engagement_rate !="--" and gender is not null;'
    cur.execute(sql)
    results = cur.fetchall()
    for i in results:
        l=[]
        l.append(i[0])
        l.append(i[1])
        l.append(i[2])
        l.append(i[3])
        data_set.append(l)
except Exception as e:
    raise e
finally:
    cur.close()
    db.close()

print(data_set)

print("==================================================")

data_set2=[]
for i in data_set:
    l=[]
    l.append(calculate_fans_num(i[0]))
    l.append(handle_rate(i[1]))
    l.append(handle_price(i[2]))
    l.append(handle_gender(i[3]))
    data_set2.append(l)

print(data_set2)

data_set2=[i for i in data_set2 if i[2]<100]        #price
data_set2=[i for i in data_set2 if i[1]<10]         #engagement
data_set2=[i for i in data_set2 if i[0]<50000]     #fans

raw_data_x=[[i[0],i[1]] for i in data_set2]
raw_data_y=[i[2] for i in data_set2]

plt.scatter([i[1] for i in data_set2],[i[2] for i in data_set2])
plt.show()

X=np.array(raw_data_x)      #(7181, 2)
Y=np.array(raw_data_y)      #(7181,)

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=666)
#
poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),     #多项式的增加特征
    ("std_scaler", StandardScaler()),           #归一化
    ("lin_reg", LinearRegression())             #线性回归
])

poly_reg.fit(X_train,y_train)
y_predict=poly_reg.predict(X_test)
print("score: ",poly_reg.score(X_test,y_test))
print("MSE: ",mean_squared_error(y_test,y_predict))


# print("=============KNN的Regressor======================")
# param_grid=[
#     {
#         'weights':['uniform'],
#         'n_neighbors':[i for i in range(1,11)]
#     },
#     {
#         'weights':['distance'],
#         'n_neighbors':[i for i in range(1,11)],
#         'p':[i for i in range(1,6)]
#     }
# ]
# knn_reg=KNeighborsRegressor()
# grid_search=GridSearchCV(knn_reg,param_grid,n_jobs=-1)
#
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)         #最好的参数{'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
#
# print("score: ",grid_search.best_estimator_.score(X_test,y_test))

