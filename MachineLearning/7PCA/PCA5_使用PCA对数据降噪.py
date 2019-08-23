import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

'''
使用PCA对数据降噪(调库实现)
'''

digits=datasets.load_digits()                               #手写数字识别
X=digits.data                                               #(1797, 64)
y=digits.target                                             #(1797,)

noisy_digits = X + np.random.normal(0, 4, size=X.shape)     #原数据加上噪音之后的数据
example_digits = noisy_digits[y==0,:][:10]                  #噪音数据中取十个图像为0的
for num in range(1,10):                                     #噪音数据中取十个图像为1-9的vstack到一起
    example_digits = np.vstack([example_digits, noisy_digits[y==num,:][:10]])

print(example_digits.shape)                                 #(100, 64)


def plot_digits(data):                                      #绘制噪音数据
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),subplot_kw={'xticks': [], 'yticks': []},gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),cmap='binary', interpolation='nearest',clim=(0, 16))
    plt.show()
plot_digits(example_digits)


print("=============PCA=======================")
pca=PCA(0.5)                                            #我们认为噪音数据的噪音比较大，所以取小一些的比例50%、
pca.fit(noisy_digits)
print("经过PCA后降到多少维: ",pca.n_components_)

print("=============降噪操作========================")
components=pca.transform(example_digits)                    #高维转低维
filtered_digits=pca.inverse_transform(components)           #低维再转回高维

plot_digits(filtered_digits)