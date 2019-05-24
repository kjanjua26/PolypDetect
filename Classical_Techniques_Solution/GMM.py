import numpy as np
import cv2
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import matplotlib.pyplot as plt

def threshold_binarize(img):
    retval2, threshold2 = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold2

img_file = "CVC-ClinicDB/Original/138.tif"
img2 = cv2.imread(img_file, 0)
img = cv2.imread(img_file)
bin_img = threshold_binarize(img2)
plt.imshow(bin_img, cmap='gray')
plt.show()
img = np.asarray(img)
height, width, _ = img.shape
#print(height, width)
newdata = bin_img.reshape(width*height, 1)
gmm = GaussianMixture(n_components=2, covariance_type='spherical', max_iter=100, n_init=1)
gmm = gmm.fit(newdata)

cluster = gmm.predict(newdata)
cluster = cluster.reshape(height, width)

'''
cluster_hook = np.zeros(shape=cluster.shape)
for x in range(cluster.shape[0]):
    for y in range(cluster.shape[1]):
        if cluster[x, y] == 1:
            cluster_hook[x, y] = 1
'''
plt.imshow(cluster, cmap='gray')
plt.show()