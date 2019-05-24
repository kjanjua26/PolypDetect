import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_binarize(img):
    return np.where(img<55, 255, 0)

def imshow_components(labels):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    plt.imshow(labeled_img, cmap='nipy_spectral')
    plt.show()

img = cv2.imread('bin_13.png', 0)
#img = threshold_binarize(img)
plt.imshow(img, cmap='gray')
plt.show()
img = img.astype('uint8')
#img = 255-img
kernel = np.ones((7, 7), np.uint8)
erosion_img = cv2.erode(img, kernel, iterations=30)
plt.imshow(erosion_img, cmap='gray')
plt.show()
vals, labels = cv2.connectedComponents(erosion_img)
imshow_components(labels)
