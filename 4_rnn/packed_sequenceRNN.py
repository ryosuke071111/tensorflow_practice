from utility import *
import tensorflow as tf

reset_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5

# static_rnn
# X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))

# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons) #セルファクトリ
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

# outputs = tf.transpose(tf.stack(output_seqs), perm=[1,0,2])

# dynamic_rnn
X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
seq_length = tf.placeholder(tf.int32,[None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)

X_batch = np.array([
        # t = 0      t = 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
seq_length_batch = np.array([2, 1, 2, 2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
  init.run()
  outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

print(outputs_val)
print(states_val )
