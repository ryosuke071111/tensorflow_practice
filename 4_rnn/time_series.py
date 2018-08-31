from utility import *
import tensorflow as tf

reset_graph()

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.OutputProjectionWrapper(
  tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
  init.run()
  for iteration in range(n_iterations):
    X_batch, y_batch = next_batch(batch_size, n_steps)
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    if iteration % 10 == 0:
      mse = loss.eval(feed_dict={X:X_batch, y:y_batch})
      print(iteration, "\tMSE:", mse)
  saver.save(sess, "./my_time_series_model")

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)


with tf.Session() as sess:
  saver.restore(sess, "./my_time_series_model")
  X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps,n_inputs)))
  y_pred = sess.run(outputs, feed_dict={X:X_new})

print(y_pred)



plt.title("testing the mopdel", fontsize=15)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("time")

save_fig("time_series_pred_plot")
plt.show()