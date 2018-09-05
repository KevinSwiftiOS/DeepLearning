import tensorflow as tf
import numpy as np
arr = np.array([[31, 23,  4, 24, 27, 34],
                [18,  3, 25,  0,  6, 35],
                [28, 14, 33, 22, 20,  8],
                [13, 30, 21, 19,  7,  9],
                [16,  1, 26, 32,  2, 29],
                [17, 12,  5, 11, 10, 15]])
#表示几维 tenorflow的输出需要加上eval
with tf.Session() as sess:
  print(tf.rank(arr).eval())
  print(tf.shape(arr).eval())
  #0表示第0 按列来 1表示按行来
  print(tf.argmax(arr, 0).eval())