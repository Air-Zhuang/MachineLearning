import numpy as np

xxx=np.arange(10)                           #一维数组
xx=np.arange(15).reshape(3,5)               #一维转成 3*5的二维数组
print(xxx);print();print(xx)

print("==============访问元素===================")
print(xx[2][3])                             #numpy中不建议这样取元素
print(xx[2,3])                              #取第三行四列的元素
print("==============切片(numpy中使用切片不同于python中的普通切片，numpy中子矩阵是原矩阵的引用，修改会影响原矩阵)===================")
print(xx[:2,:3])                            #矩阵切片，前两行，前三列，是原矩阵的引用
print(xx[:2,:3].copy())                     #生成全新的子矩阵
print(xx[:2,::2])                           #取前两行，每行步长2取数
print(xx[::-1,::-1])                        #实现了矩阵左右翻转
print("==============取某一行或某一列，实现了降维处理===================")
print(xx[0,:])                              #取某一行，实现了降维处理
print(xx[0,])                               #同上
print(xx[0])                                #同上
print(xx[:,0])                              #取第1列(也用于将列向量转换成行向量)
print("==============reshape()===================")
print(xxx.reshape(2,5))                       #转为2维数组，2*5
print(xxx.reshape(2,-1))                      #转为2维数组，2行，列自动计算
print(xxx.reshape(-1,5))                      #转为2维数组，5列，行自动计算

print("==============合并操作(相同维度)===================")
x=np.array([1,2,3])
y=np.array([3,2,1])
z=np.array([666,666,666])
A=np.array([[1,2,3],[4,5,6]])
B=np.array([[2,2],[2,2]])

print(np.concatenate([x,y,z]))                  #一维数组合并
print(np.concatenate([A,A]))                    #二维数组合并
print(np.concatenate([A,A],axis=1))             #二维数组合并,按照行拼接
print("==============合并操作(不同维度)===================")
print(np.concatenate([A,z.reshape(1,-1)]))      #一维二维合并(先转换成相同形状)，不建议
print(np.vstack([A,z]))                         #一维二维垂直合并，建议使用这种
print(np.vstack([A]*A.shape[0]))                #一维二维垂直合并，建议使用这种
print(np.hstack([A,B]))                         #一维二维水平合并，建议使用这种
print("==============向量堆叠===================")
xl=np.array([2,3])
print(np.tile(xl,(2,3)))                        #纵向叠两次，横向叠三次

print("==============分割操作===================")
a=np.arange(10)
b=np.arange(16).reshape((4,4))
print(b)
print("==============分割一维===================")
x1,x2,x3=np.split(a,[3,7])                      #切三段0-2,3-6,7-9
print(x1,x2,x3)
x1,x2=np.split(a,[5])                           #切两端
print(x1,x2)
print("==============分割二维===================")
x1,x2=np.split(b,[2])                           #上下分割
print(x1)
print(x2)
x1,x2=np.split(b,[2],axis=1)                    #左右分割
print(x1)
print(x2)
x1,x2=np.vsplit(b,[2])                          #上下分割
print(x1)
print(x2)
x1,x2=np.hsplit(b,[2])                          #左右分割
print(x1)
print(x2)
print("==============分割二维最后一列，并转换成向量===================")
x,y=np.hsplit(b,[-1])
print(y)
print(y[:,0])

