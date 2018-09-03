from utility import *
import tensorflow as tf
import math

mnist = input_data.read_data_sets("/tmp/data/")


reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

X_test = mnist.test.images.reshape((-1, n_steps, n_inputs)) #-1のイメージがわかない
y_test = mnist.test.labels

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32,[None])

#深層RNNを作るときの準備のコード
#-----------------------------
lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons)
          for layer in range(n_layers)]

multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
#-------------------------------
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

top_layer_h_state = states[-1][1]

logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name = "loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
  init.run()
  for epoch in range(n_epochs):
    X_batch, y_batch = mnist.train.next_batch(batch_size)
    X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
    acc_test = accuracy.eval(feed_dict={X:X_test, y:y_test})
    print("Epoch", epoch, "Train accuracy=", acc_train, "Test accuracy=", acc_test)

pi = math.pi
x = np.linspace(0, 2*pi, 100)  #0から2πまでの範囲を100分割したnumpy配列
y = np.sin(x)


plt.title("LSTM experiments", fontsize =14)
# plt.plot([0, 0, 0,20], "k-")
plt.plot(x)
plt.plot(y)
plt.xlabel("time")
plt.axis([-1, 10, 0, 1])
plt.show()


