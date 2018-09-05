import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist      = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST loaded")
print (trainimg.shape)
print (trainlabel.shape)
print (testimg.shape)
print (testlabel.shape)
#print (trainimg)
print (trainlabel[0])

#进行模型的搭建 None 表示无穷值 是 28 * 28的维度
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])
W = tf.Variable(tf.random_uniform([784,10],-1.0,1.0))
b = tf.Variable(tf.zeros([10]))
#逻辑斯蒂回归模型
actv = tf.nn.softmax(tf.matmul(x,W) + b)
#损失函数 recuction_indices 表示处理的维度 0表示列 0表示维度
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv),reduction_indices=1))

#优化
train = tf.train.GradientDescentOptimizer(0.01)
optm = train.minimize(cost)

#进行预测 argmax第二个表示第几维
pred = tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
#预测准确度 将true false 转化为float
accr = tf.reduce_mean(tf.cast(pred,'float'))

init = tf.global_variables_initializer()

#batch迭代几次
traing_epochs = 50
#每个batch的大小
batch_size = 100
display_step = 5
sess = tf.Session()
sess.run(init)
# MINI-BATCH LEARNING
for epoch in range(traing_epochs):
    avg_cost = 0.
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
    # DISPLAY
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
               % (epoch, traing_epochs, avg_cost, train_acc, test_acc))
print ("DONE")