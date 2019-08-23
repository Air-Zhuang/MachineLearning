import numpy as np

print("==============基础矩阵和属性===========")
lst=[[1,3,5],[2,4,6]]
np_lst=np.array(lst)                            #定义一个array
np_lst2=np.array(lst,dtype=np.float)            #定义一个array，指定类型bool,int,int8/16/32/64/128,uint,uint8/16/32/64/128,float,float 16/32/64,complex64/128
print(np_lst)
print(type(np_lst))
print(np_lst2)
print(np_lst2.shape)                            #几行几列
print(np_lst2.ndim)                             #数组维度
print(np_lst2.dtype)                            #数据类型
print(np_lst2.itemsize)                         #每个元素所占的字节数
print(np_lst2.size)                             #元素个数

print("==============0矩阵===================")
print(np.zeros([2,4]))

print("==============1矩阵===================")
print(np.ones([3,5]))

print("==============单位矩阵===================")
print(np.eye(4))

print("==============元素一样的矩阵===================")
print(np.full([3,5],666))
print(np.full(10,666))
print(np.full([10],666))

print("==============随机数矩阵===================")
print(np.random.rand(2,4))                              #随机数矩阵
print("==============符合标准正态分布的随机数组===================")
print(np.random.randn(10))                              #创建长度为10的符合标准正态分布的随机数组
print("==============0-1之间随机浮点数矩阵===================")
print(np.random.random((2,4)))                          #0-1之间随机浮点数
print("==============随机整数矩阵===================")
print(np.random.randint(1,10,size=10))                  #取不到10
print(np.random.randint(1,10,size=[2,3]))               #生成一个随机整数矩阵
np.random.seed(666)                                     #指定随机种子之后随机矩阵会相同
print(np.random.randint(1,10,size=10))
np.random.seed(666)
print(np.random.randint(1,10,size=10))
print("==============其他随机矩阵===================")
print(np.random.normal())                               #正态分布均值为0，方差为1的随机浮点数
print(np.random.normal(10,100))                         #均值10，方差100
print(np.random.normal(0,1,(2,3)))                      #均值0，方差1,2*3矩阵
print(np.random.choice([10,20,30,40]))                  #从指定数字中获取随机数
print(np.random.beta(1,100,10))                         #beta分布,氛围1-100,生成10个
print()

print("==============矩阵唯一值一维矩阵===================")
r=np.random.randint(10,size=20).reshape(4,5)
print(np.unique(r))

print("==============arange(相当于range)===================")
print(np.arange(0,1,0.2))                       #这里的步长可以为浮点数
print(np.arange(10))
print()

print("==============arange(等长截出)===================")
print(np.linspace(0,20,10))                     #在0,20之间等长截出10个数，包含0和20
print(np.linspace(0,20,11))
print()

print("==============array存入文件，读取文件===================")
# x=np.arange(12).reshape(3,4)
# y=np.arange(20).reshape(4,5)
# np.save("one_array",x)                                  #存成 one_array.npy 格式的文件
# np.load("one_array.npy")                                #读取array
#
# np.savez("two_array.npz",a=x,b=y)                       #存两个array
# c=np.load("two_array.npz")                              #读取
# print(c['a'],c['b'])