import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

'''
PCA求特征脸(调库实现)(需要下载数据)
'''

faces=fetch_lfw_people()

print(faces.data)                   #(13233,2914)
print(faces.images.shape)           #(13233,62,47)

random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36,:]            #(36,2914)  随机选36张脸


def plot_faces(faces):              #绘制
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),subplot_kw={'xticks': [], 'yticks': []},gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()
plot_faces(example_faces)


print("============PCA求特征脸===============")
pca = PCA(svd_solver='randomized')  #使用随机的方式求PCA(快)
pca.fit(X)

print(pca.components_.shape)        #主成分

plot_faces(pca.components_[:36,:])  #主成分绘制特征脸