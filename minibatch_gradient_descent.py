from utility import *
from tensorflowrun import reset_graph
from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from linear_regression import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


n_epochs = 1000
learning_rate = 0.01

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n+1), name = "X")
y = tf.placeholder(tf.float32, shape=(None, 1), name = "y")

theta = tf.Variable(tf.random_uniform([n+1, n], -1.0, 1.0, seed = 42), name ="theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 100
n_batchs = int(np.ceil(m/batch_size))

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    for batch_index in range(n_batchs):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_index, n_batchs)
      sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
  best_theta = theta.eval()

with tf.Session()as sess:
  save.restore(sees, "/tensorflow_practice/my_model_final.meta")
  best_theta_restored = theta.eval()

np.allclose(best_theta, best_theta_restored)

saver = tf.train.Saver({"weights":theta})

# reset_graph()
# saver = tf.train.import_meta_graph()
