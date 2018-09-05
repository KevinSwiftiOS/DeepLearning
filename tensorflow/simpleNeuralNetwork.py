import  numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data/', one_hot=True)
#进行网络层的定义
#第一层神经元有256个 第二层神经元有128个 输入与输出
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])

#权重的初始化
stddev = 0.1
weights = {
    "w1":tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev = stddev)),
    "w2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_classes],stddev=stddev))
}
bias = {
    "b1":tf.Variable(tf.zeros([n_hidden_1])),
    "b2":tf.Variable(tf.zeros([n_hidden_2])),
    "out":tf.Variable(tf.zeros([n_classes]))
}
#前向传播
def forward_pro(x,weights,bias):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x , weights["w1"]) ,bias["b1"]))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,weights["w2"]) , bias["b2"]))
    return tf.add(tf.matmul(layer_2,weights["out"]) , bias["out"])

#定义损失函数
pred = forward_pro(x,weights,bias)
#交叉熵作为损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
#学习率
learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate)
optm = train.minimize(cost)
#表示准确值
correct = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(correct,"float"))
#全部变量进行初始化
init = tf.global_variables_initializer()
print ("FUNCTIONS READY")
train_epoch = 50
batch_size = 100
display_step = 4
sess = tf.Session()
sess.run(init)
for epoch in range(train_epoch):
    avg_cost = 0
    num_epoch = int(mnist.train.num_examples / batch_size)
    for i in range(num_epoch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        feeds = {
            x:batch_xs,y:batch_ys
        }
        sess.run(optm,feed_dict=feeds)
        #总体的损失值相加
        avg_cost += sess.run(cost,feed_dict=feeds)
    avg_cost = avg_cost / num_epoch
    #进行输出
    if (epoch + 1) % display_step == 0:
        print("Epoch : %03d/%03d cost %.9f" %(epoch,train_epoch,avg_cost))
        feeds = {x:batch_xs,y:batch_ys}
        #训练误差与测试误差
        train_accr = sess.run(accr,feed_dict=feeds)
        print("TEST ACCURACY: %.3f" % (train_accr))
        feeds = {x:mnist.test.images,y:mnist.test.labels}
        test_accr = sess.run(accr,feed_dict=feeds)
        print("TEST ACCURACY: %.3f" % (test_accr))
print("Done")
