import tensorflow as tf
import cv2
import network
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
import numpy as np

X = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="image")
img_file = '1ETIS.tif'
img = cv2.imread(img_file)
img = cv2.resize(img, (256, 256))
pred = network.make_unet(X, False)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model-epoch500.ckpt")
    mask_pred = sess.run(pred, feed_dict={X:[img]})
    mask_pred = np.squeeze(mask_pred)
    mask_pred = mask_pred > 0.9999
    plt.imshow(mask_pred, cmap='gray')
    plt.show()
    labeled_heatmap, n_labels = label(mask_pred)
    bbox = []
    for i in range(n_labels):
        mask_i = labeled_heatmap == (i + 1)
        nonzero = np.nonzero(mask_i)
        nonzero_row = nonzero[0]
        nonzero_col = nonzero[1]
        left_top = min(nonzero_col), min(nonzero_row)
        right_bot = max(nonzero_col), max(nonzero_row)
        image = cv2.rectangle(img, left_top, right_bot, color=(0, 255, 0), thickness=2)      
        bbox.append((left_top, right_bot))
    plt.imshow(img, cmap='gray')
    plt.show()