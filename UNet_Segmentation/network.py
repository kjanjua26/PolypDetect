import tensorflow as tf

def conv_conv_pool(input_, n_filters, training, flags, name, pool=True, activation=tf.nn.relu):
    net = input_
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net
        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
        return net, pool

def upconv_concat(inputA, input_B, n_filter, flags, name):
    up_conv = upconv_2D(inputA, n_filter, flags, name)
    return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))

def upconv_2D(tensor, n_filter, flags, name):
    return tf.layers.conv2d_transpose(tensor, filters=n_filter, kernel_size=2, strides=2, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1), name="upsample_{}".format(name))

def make_unet(X, training, flags=None):
    net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(pool4, [128, 128], training, flags, name=5, pool=False)
    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)
    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)
    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)
    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)
    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')


def IOU_(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + 1e-7
    return tf.reduce_mean(intersection / union)