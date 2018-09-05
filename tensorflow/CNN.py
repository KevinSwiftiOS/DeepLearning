import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#进行数据的读写
mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print ("MNIST ready")

#输入输出的超参数的定义
n_inputs = 784
n_outputs = 10
weights = {
    #卷积层的定义 表示为3 * 3 1的深度 有64个filter
    "wc1":tf.Variable(tf.random_normal([3,3,1,64],stddev=0.1)),
     #第一个特征有64 所以深度为64
    "wc2":tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01)),
    #第一个全连接层  7 * 7 的 128的深度
    "wd1":tf.Variable(tf.random_normal([7 * 7 * 128,1024],stddev=0.01)),
    "wd2":tf.Variable(tf.random_normal([1024,n_outputs],stddev=0.01))

}

bias = {
    "bc1":tf.Variable(tf.zeros([64])),
   'bc2': tf.Variable(tf.zeros([128])),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_outputs], stddev=0.1))

}
#前项传播的编写
def conv_basic(_input,_w,_b,_keeppratio):
    """

    :param _input: 表示输入的图片
    :param _w: 权重
    :param _b: 参数
    :param _keeppratio: 预留下多少神经 在dropout时使用
    :return:
    """
    #必须转换为卷积支持的4维类型 28*28的像素 深度为1 前面的batch_size自己推到
    _input_r = tf.reshape(_input,shape = [-1,28,28,1])
    #第一层卷积层 stride4个参数表示分别在batch_size上移动 w h上移动 和在深度上的移动 第一和最后一般为1 same表示没有的时候用0自动填充 另一种
    #valud表示忽略
    _conv1 = tf.nn.conv2d(_input_r,_w['wc1'],strides=[1,1,1,1],padding='SAME')
    #激活函数
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1,_b["bc1"]))
    #做池化 ksize与上式表示相同2 2表示大小 1 1表示滑动 stride基本与ksize相同
    _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #做dropout
    _pool_dr1 = tf.nn.dropout(_pool1,_keeppratio)


    #做第二层的卷积与池化
    _conv2 = tf.nn.conv2d(_pool_dr1,_w["wc2"],strides=[1,1,1,1],padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b["bc2"]))
    _pool2 = tf.nn.max_pool(_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #做dropout

    _pool_dr2 = tf.nn.dropout(_pool2,_keeppratio)
    #做全连接层的转置
    _densel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel,_w["wd1"]),_b["bd1"]))
    _fc1_dr = tf.nn.dropout(_fc1,_keeppratio)
    _out = tf.add(tf.matmul(_fc1_dr,_w["wd2"]),_b["bd2"])
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _densel,
           'fc1': _fc1, 'fc_dr1': _fc1_dr, 'out': _out
           }
    return out


#做输入与输出和keepprob的站位
x = tf.placeholder(tf.float32,[None,n_inputs])
y = tf.placeholder(tf.float32,[None,n_outputs])
keepratio = tf.placeholder(tf.float32)

#进行预测
_pred = conv_basic(x,weights,bias,keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))
optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
correct = tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(correct,'float'))
init = tf.global_variables_initializer()











#进行模型的保存
do_train = 0 #表示当前是训练还是测试
save_step = 1
saver = tf.train.Saver(max_to_keep=3) #表示最多保存3个模型
#因为一个模型的大小为6 7十兆 所以只需保存一个即可
sess = tf.Session()
sess.run(init)











train_epoch = 15
batch_size = 16
display_step = 1
if do_train == 1:
 for epoch in range(train_epoch):
    avg_cost = 0.
    #total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = 10
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, train_epoch, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        #test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
        #pint (" Test accuracy: %.3f" % (test_acc))
    #进行保存
    if epoch % save_step == 0:
        saver.save(sess,"save/nes/cnn_mnist_basic.ckpt" + str
        (epoch))


#进行测试
if(do_train == 0):
    epoch = train_epoch - 1
    saver.restore(sess,"save/nes/cnn_mnist_basic.ckpt" + str
        (epoch))
    test_acc = sess.run(accr,feed_dict={
        x:testimg,
        y:testlabel,
        keepratio:1
    })
    print(" TEST ACCURACY: %.3f" % (test_acc))

