import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn_cell,rnn
#进行数据集的下载
mnist = input_data.read_data_sets("data/",one_hot=True)
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print ("MNIST loaded")
#神经元层数的定义
diminput = 28
dimhidden = 128
dimoutput = nclasses
nsteps = 28
weights = {
    "hidden":tf.Variable(tf.random_normal([diminput,dimhidden],stddev= 0.01)),
    "out":tf.Variable(tf.random_normal([dimhidden,dimoutput]))

}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}
#RNN的定义
def _RNN(_X,_W,_b,_nsteps,_name):
    #将前两个进行专职
#将[batchsize, nsteps, diminput] 转化为  [nsteps*batchsize, diminput]
  _X = tf.transpose(_X,[1,0,2])
  _X = tf.reshape(_X,[-1,diminput])
  _H = tf.add(tf.matmul(_X,_W['hidden']),_b['hidden'])
  _H_split = tf.split(_H,_nsteps,0)
  with tf.variable_scope(_name) as scope:
      #重复利用 防止命名冲突
      scope.reuse_variables()
      #隐藏层的
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
          dimhidden,forget_bias=1.0
      )
      LSTM_0, LSTM_S = tf.nn.static_rnn(lstm_cell, _H_split, dtype=tf.float32)
      _O = tf.matmul(LSTM_0[-1],_W['out']) + _b['out']
      return {
          'X': _X, 'H': _H, 'Hsplit': _H_split,
          'LSTM_O': LSTM_0, 'LSTM_S': LSTM_S, 'O': _O
      }
print("Network ready")
learning_rate = 0.001
x = tf.placeholder(tf.float32,[None,nsteps,diminput])
y = tf.placeholder(tf.float32,[None,dimoutput])
#RNN的使用
myrnn = _RNN(x,weights,biases,nsteps,'basic')
#进行预测的值
pred = myrnn['0']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),'float'))
init = tf.global_variables_initializer()
print("ready")
#进行训练
train_epoch = 5
batch_size = 16
display_step = 1
sess = tf.Session()
sess.run(init)
for epoch in range(train_epoch):
    avg_cost = 0
    total_batch = 100
    for i in range(total_batch):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size,nsteps,diminput))
        feeds = {
            x:batch_x,
            y:batch_y
        }
        sess.run(optm,feed_dict=feeds)
        avg_cost += sess.run(cost,feed_dict=feeds) / total_batch

        if epoch % display_step == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, train_epoch, avg_cost))
            feeds = {x: batch_x, y: batch_y}
            train_acc = sess.run(accr, feed_dict=feeds)
            print(" Training accuracy: %.3f" % (train_acc))
            testimgs = testimgs.reshape(ntest,nsteps,diminput)
            feeds = {x: testimgs, y: testlabels}
            test_acc = sess.run(accr, feed_dict=feeds)
            print(" Test accuracy: %.3f" % (test_acc))
