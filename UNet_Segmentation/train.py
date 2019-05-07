import tensorflow as tf
import network
import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
NUM_EPOCHS = 500
BATCH_SIZE = 32

X = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="image")
Y = tf.placeholder(tf.float32, shape=[None, 256, 256, 1], name="mask")
isTraining = tf.placeholder(tf.bool, name="iftraining")
global_step = tf.Variable(0, trainable=False, name='global_step')

def build_network(input_images, mask_labels):
    logits = network.make_unet(input_images, isTraining)
    loss = 10-network.IOU_(logits, mask_labels)
    return loss
  
def train(x_train, x_val, y_train, y_val):
    loss = build_network(X, Y)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(1, NUM_EPOCHS+1):
        batch_images, batch_labels = utils.next_batch(BATCH_SIZE, x_train, y_train)
        _, train_iou = sess.run([train_op, loss], feed_dict={X: batch_images, Y: batch_labels, isTraining: True})
        print('Step: {} Loss: {}'.format(epoch, train_iou))
        if epoch % 100 == 0:
            val_iou = sess.run([loss], feed_dict={X: x_val, Y: y_val, isTraining: False})
            print("")
            print("Step: {} Loss: {}".format(epoch, val_iou[0]))
            save_path = saver.save(sess, "model/model-epoch{}.ckpt".format(epoch))
            print("Model saved for epoch # {}".format(epoch))
            print("")

if __name__ == "__main__":
    x_train, x_val, y_train, y_val = utils.train_test_split_data('CVC-ClinicDB')
    train(x_train, x_val, y_train, y_val)
