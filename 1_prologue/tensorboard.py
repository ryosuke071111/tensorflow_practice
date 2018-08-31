from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utility import *

#TensorBoard用
#-------------------------------------------------
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run - {}/".format(root_logdir, now)
#--------------------------------------------------

n_epochs = 1000
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=(None, n+1), name = "X")
y = tf.placeholder(tf.float32, shape=(None, 1), name = "y")

theta = tf.Variable(tf.random_uniform([n+1, n], -1.0, 1.0, seed = 42), name ="theta")
y_pred = tf.matmul(X, theta, name = "predictions")

with tf.name_scope("loss") as scope:
  error = y_pred - y
  mse = tf.reduce_mean(tf.square(error), name = "mse")



optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

#TensorBard用
#------------------------------------------------------------------
mse_summary = tf.summary.scalar('MSE',mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#------------------------------------------------------------------

n_epochs = 10
batch_size = 100
n_batchs = int(np.ceil(m/batch_size))

## モデル/重み/インデックスの保存
with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    for batch_index in range(n_batchs):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_index, n_batchs)
      if batch_index % 10 == 0:
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y:y_batch})
        step = epoch * n_batchs + batch_index
        file_writer.add_summary(summary_str, step)
      sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

  best_theta = theta.eval()


file_writer.close()

