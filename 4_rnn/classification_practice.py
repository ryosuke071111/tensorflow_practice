from utility import *
import tensorflow as tf
import math

reset_graph()

n_steps = 40
n_inputs = 40
n_neurons = 100
n_outputs = 2

lstm_cells = tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons)
learning_rate = 0.001

X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

outputs, states = tf.nn.dynamic_rnn(lstm_cells, X, dtype=tf.float32)

logits = tf.layers.dense(X, n_outputs, name = "softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name = "loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correc, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
  init.run()
  for epoch in range(n_epochs):
    X_batch, y_batch =
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
    acc_test = accuracy.eval(feed_dict={X:X_test, y:y_test})

