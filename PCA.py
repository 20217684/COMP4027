import matplotlib.pyplot as plt

import pandas as pd
mnist = pd.read_csv('mnist.csv')
mnist.head()
#Head( ) gives the first 5 data points of the data.
mnist.drop(columns='label', inplace=True)
mnist.head()
print(mnist.shape)
#(60000, 784)

second_image = mnist.iloc[1].values.reshape([28,28])
plt.imshow(second_image, cmap='gray_r')
plt.title('Second image: Digit 0', fontsize=15, pad=15)
plt.savefig("Second image.png")
plt.cla()


#2nd image
print(mnist.iloc[1].min())
print(mnist.iloc[1].max())
#0
#255


import numpy as np
from sklearn.decomposition import PCA

pca_784 = PCA(n_components=784)
pca_784.fit(mnist)

plt.grid()
plt.plot(np.cumsum(pca_784.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('Scree plot.png')
plt.cla()



pca_3 = PCA(n_components=3)
mnist_pca_3_reduced = pca_3.fit_transform(mnist)
mnist_pca_3_recovered = pca_3.inverse_transform(mnist_pca_3_reduced)

image_pca_3 = mnist_pca_3_recovered[1,:].reshape([28,28])
plt.imshow(image_pca_3, cmap='gray_r')
plt.title('Compressed image with 3 components', fontsize=15, pad=15)
plt.savefig("image_pca_3.png")
plt.cla()



pca_10 = PCA(n_components=10)
mnist_pca_10_reduced = pca_10.fit_transform(mnist)
mnist_pca_10_recovered = pca_10.inverse_transform(mnist_pca_10_reduced)

image_pca_10 = mnist_pca_10_recovered[1,:].reshape([28,28])
plt.imshow(image_pca_10, cmap='gray_r')
plt.title('Compressed image with 10 components', fontsize=15, pad=15)
plt.savefig("image_pca_10.png")
plt.cla()


pca_50 = PCA(n_components=50)
mnist_pca_50_reduced = pca_50.fit_transform(mnist)
mnist_pca_50_recovered = pca_50.inverse_transform(mnist_pca_50_reduced)

image_pca_50 = mnist_pca_50_recovered[1,:].reshape([28,28])
plt.imshow(image_pca_50, cmap='gray_r')
plt.title('Compressed image with 50 components', fontsize=15, pad=15)
plt.savefig("image_pca_50.png")


pca_184 = PCA(n_components=184)
mnist_pca_184_reduced = pca_184.fit_transform(mnist)
mnist_pca_184_recovered = pca_184.inverse_transform(mnist_pca_184_reduced)

image_pca_184 = mnist_pca_184_recovered[1,:].reshape([28,28])
plt.imshow(image_pca_184, cmap='gray_r')
plt.title('Compressed image with 184 components', fontsize=15, pad=15)
plt.savefig("image_pca_184.png")

T= np.cumsum(pca_184.explained_variance_ratio_ * 100)[-1]
#96.11980535398752
print("variability =",round(T,2),"%")
#96.12%