from tensorflowrun import reset_graph
from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf


reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m ,1)),housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name = "X")
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name = "X")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT),Y)

with tf.Session() as sees:
  theta_values = theta.eval()

print(theta_values)

