from tensorflowrun import reset_graph
from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from linear_regression import *

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

reset_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32, name = "y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0,seed=42),name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")

#手動微分
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
#手動トレーニング
# training_op = tf.assign(theta, theta -learning_rate*gradients)

#自動微分
gradients = tf.gradients(mse,[theta])[0]

#自動トレーニング(GradientDescentOptimizer/MomentumOptimizer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch % 100 == 0:
      print("Epoch", epoch, "MSE=", mse.eval())
    sess.run(training_op)
  best_theta = theta.eval()

print(best_theta)
