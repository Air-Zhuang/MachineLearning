import numpy as np

print("==============索引=================")
x=np.random.normal(0,1,1000)                    #正态分布均值为0，方差为1的1000个
print(np.argmin(x))                             #最小值的位置索引
print(np.argmax(x))                             #最大值的位置索引

print("==============排序和使用索引=================")
x=np.random.randint(1,10,size=[4,4])
print(x)
print(np.sort(x))                               #排序,默认axis参数为1，每行排序
print(np.sort(x,axis=0))                        #排序,每列排序
xx=np.arange(10)
np.random.shuffle(xx)
print(xx)
print(np.argsort(xx))                           #索引排序
print(np.partition(xx,4))                       #快速排序中的子过程，找一个标定点，左边的都比它小，右边的都比它大
print(np.argpartition(xx,4))

print("==============Fancy Indexing(高级索引)=================")
print("==============一维=================")
x=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
ind=[3,5,8]
print(x[ind])                                   #生成向量
ind2=np.array([[0,1],[1,3]])                    #ind可以为数组
print(x[ind2])                                  #生成矩阵
print("==============二维=================")
X=x.reshape(4,-1)
print(X)
print()
row=np.array([0,1,2])
col=np.array([1,2,3])
print(X[row,col])                               #生成向量
print(X[0,col])                                 #生成向量
print(X[:2,col])                                #生成矩阵
col2=[True,False,True,True]
print(X[:2,col2])                               #生成矩阵

print("==============比较=================")
print(x)
print(x<3)                                      #每个元素与3比较大小
print(x==3)
print(X<6)                                      #二维矩阵比较
print("==============判断=================")
print("==============一维=================")
print(np.sum(x<=3))                             #小于三的个数
print(np.sum((x>3) & (x<6)))                    #大于三且小于6的个数,(这里不能用and or 要用位运算符)
print(np.sum((x%2==0) | (x>10)))                #偶数或大于10的个数
print(np.sum(~(x==0)))                          #非0个数
print(np.count_nonzero(x<=3))                   #判断非0个数，这里传入布尔数组，表示求小于三的个数
print(np.any(x==0))                             #是否有值为0的元素
print(np.all(x>=0))                             #是否所有元素都大于等于0
print("==============二维=================")
print(np.sum(X%2==0))                           #偶数个数
print(np.sum(X%2==0,axis=0))                    #沿着行的维度，(每一列)
print(np.all(X>0,axis=1))                       #沿着列的维度，(每一行)

print(X[X[:,3]%3==0,:])                         #求每一行的第四列能被3整除的行
