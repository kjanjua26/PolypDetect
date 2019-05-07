import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="image")
img_file = 'CVC-ClinicDB/Ground Truth/66.tif'
img = cv2.imread(img_file, 0)
img = cv2.resize(img, (256, 256))
plt.imshow(img, cmap='gray')
plt.show()
img = np.reshape(img, (256, 256, 1))
pred = make_unet(X, False)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("model")
    saver.restore(sess, "model/model-epoch600.ckpt")
    mask_pred = sess.run(pred, feed_dict={X:[img]})
    mask_pred = np.squeeze(mask_pred)
    plt.imshow(mask_pred, cmap='gray')
    plt.show()
