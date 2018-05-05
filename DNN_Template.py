import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=None, name='X')
y = tf.placeholder(tf.int64, shape=None, name='y')

with tf.name_scope('dnn"'):

with tf.name_scope('loss'):

with tf.name_scope('train'):

with tf.name_scope('eval'):

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = None
n_batches = None
batch_size = None

best_epoch = None
best_acc = None
best_model = None
early_stop_count = 10

with tf.Session() as sess:
    init.run()
    for epoch in range (n_epochs):
        early_stop_count = early_stop_count + 1
        for iterations in range (X.shape[1] // batch_size):
            X_batch, y_batch = next_batch (X_train_scaled, y_train, batch_size, iteration)
            sess.run(training_op, feed_dict={X:X_batch , y:y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test_scaled, y: y_test})
        acc_val = accuracy.eval(feed_dict={X: X_val, y: y_val})
        if (acc_val > best_acc):
            best_acc = acc_val
            best_epoch = epoch
            early_stop_count = 0
            best_model = saver.save(sess, './my_best_model.ckpt')
        if (early_stop_count > 10):
            break
        print(str(acc_train) + "..." + str(acc_test) + "..." + str(acc_val))

    final_model = saver.save(sess, './my_final_model.ckpt')


