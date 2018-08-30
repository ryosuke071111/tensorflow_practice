# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import os
import tensorflow as tf

def reset_graph(seed = 42):
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  np.random.seed(seed)

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x*x*y+y+2

# × 個々の変数を個別に初期化
with tf.Session() as sess:
  x.initializer.run()
  y.initializer.run()
  result = f.eval()

# × 全ての変数を初期化するノードをグラフ内に作成
init = tf.global_variables_initializer()
with  tf.Session() as sees:
  init.run()
  result = f.eval()

# ○ withブロックなしでデフォルトでセッションを実行できる
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()

print(result)
sess.close()

reset_graph()

#作成したノードは自動的にデフォルトグラフになる
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()


#独立した別のノードをwtihブロックで保存する
graph = tf.Graph()
with graph.as_default():
  x2 = tf.Variable(2)
x2.graph is tf.get_default_graph()

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

# × wとxが二回読み込まれるので非効率
with tf.Session() as sess:
  print(y.eval())
  print(z.eval())

# ○ 一回の実行で両方を読ませる
with tf.Session() as sees:
  y_val, z_val = sess.run([y, z])
  print(y_val)
  print(z_val)
