import tensorflow as tf

def make_unet(X, training):
    input_layer = X / 127.5 - 1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=8, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_1")
    bn1 = tf.layers.batch_normalization(conv1, training=training, name="bn_1")
    pool1 = tf.layers.max_pooling2d(bn1, (2, 2), strides=(2, 2), name="pool_1")
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_2")
    bn2 = tf.layers.batch_normalization(conv2, training=training, name="bn_2")
    pool2 = tf.layers.max_pooling2d(bn2, (2, 2), strides=(2, 2), name="pool_2")
    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_3")
    bn3 = tf.layers.batch_normalization(conv3, training=training, name="bn_3")
    pool3 = tf.layers.max_pooling2d(bn3, (2, 2), strides=(2, 2), name="pool_3")
    conv4 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_4")
    bn4 = tf.layers.batch_normalization(conv4, training=training, name="bn_4")
    pool4 = tf.layers.max_pooling2d(bn4, (2, 2), strides=(2, 2), name="pool_4")
    conv5 = tf.layers.conv2d(inputs=pool4, filters=128, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_5")
    bn5 = tf.layers.batch_normalization(conv5, training=training, name="bn_5")
    up_conv6 = tf.layers.conv2d_transpose(bn5, filters=64, kernel_size=2, strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="upconv_6")
    concat6 = tf.concat([up_conv6, conv4], axis=-1, name="concat_6")
    conv6 = tf.layers.conv2d(inputs=concat6, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_6") 
    bn6 = tf.layers.batch_normalization(conv6, training=training, name="bn_6")
    up_conv7 = tf.layers.conv2d_transpose(bn6, filters=32, kernel_size=2, strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="upconv_7")
    concat7 = tf.concat([up_conv7,conv3], axis=-1, name="concat_7")
    conv7 = tf.layers.conv2d(inputs=concat7, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_7")
    bn7 = tf.layers.batch_normalization(conv7, training=training, name="bn_7")
    up_conv8 = tf.layers.conv2d_transpose(bn7, filters=16, kernel_size=2, strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="upconv_8")
    concat8 = tf.concat([up_conv8, conv2], axis=-1, name="concat_8")
    conv8 = tf.layers.conv2d(inputs=concat8, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_8")
    bn8 = tf.layers.batch_normalization(conv8, training=training, name="bn_8")
    up_conv9 = tf.layers.conv2d_transpose(bn8, filters=8, kernel_size=2, strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="upconv_9")
    concat9 = tf.concat([up_conv9, conv1], axis=-1, name="concat_9")
    conv9 = tf.layers.conv2d(inputs=concat9, filters=32, kernel_size=[5, 5],padding="same",activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_layer_9")
    bn9 = tf.layers.batch_normalization(conv9, training=training, name="bn_9")
    out = tf.layers.conv2d(bn9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')
    return out

def IOU_(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])
    intersection = 10*tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
    return tf.reduce_mean(intersection / union)
