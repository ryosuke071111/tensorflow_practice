# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# Common imports
import numpy as np
import os
import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x*x*y+y+2

#個々の変数を個別に初期化
with tf.Session() as sess:
  x.initializer.run()
  y.initializer.run()
  result = f.eval()

#全ての変数を初期化するノードをグラフ内に作成
init = tf.global_variables_initializer()
with  tf.Session() as sees:
  init.run()
  result = f.eval()

# withブロックなしでデフォルトでセッションを実行できる
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()

print(result)
sess.close()
