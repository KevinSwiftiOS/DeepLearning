import tensorflow as tf
#变量 1 * 2
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
#初始化
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #进行执行
    print(y.eval())


#一些常规基本操作 float32的类型
#tf.zeros([3, 4], int32)
tensor = tf.constant([1,2,3,4,5,6])
print(tensor)
tensor = tf.constant(-1.0,shape = [2,3])
print(tensor)
#linespace 进行切割 在一个范围内有几个数
print(tf.linspace(10.0,12.0,3,name = 'linspace'))
 #平均值为-1 方差为4  随机值初始化
norm = tf.random_normal([2,3],mean = -1,stddev = 4)
print(111)
print(norm)
c = tf.constant([[1,2],[3,4],[5,6]])
shuff = tf.random_shuffle(c)
sess = tf.Session()
print(222)
print(sess.run(norm))
#随意变换值的顺序
print(sess.run(shuff))
#进行+1的操作
state = tf.Variable(0)
#进行相加
new_value = tf.add(state,tf.constant(1))
#进行赋值
update = tf.assign(state,new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for i in range(3):
       print(sess.run(update))
       print(sess.run(state))



#训练好的值进行保存
w = tf.Variable([[0.5,1.0]])
x = tf.Variable([[2.0],[1.0]])
y = tf.matmul(w,x)
init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(y)
#     save_path = saver.save(sess,"E://tensorflow")
#     print("Model saved in file:",save_path)

#将numpy转化为tensorflow类型
import numpy as np
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(ta))

#placeholder占据了一个占位符 通过feed来进行喂养和填充
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.matmul(input1,input2)
with tf.Session() as sess:
    print(sess.run([output],
                   feed_dict={
                       input1:[[7,2]],
                       input2:[[2],[1]]
                   }))
