import tensorflow as tf

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])
sub = tf.subtract(x,a)
init = tf.global_variables_initializer()
#使用初始化
with tf.Session() as sess:
    #表示指派特定的gpu来进行操作
    with tf.device("/gpu:1"):
        sess.run(init)
        result = sess.run(sub)
        print(result)

g = tf.Graph()
with g.as_default():
    c = tf.constant(30.0)
    print(c.graph is g)
# 一个图中包含有一个名称范围的堆栈，在使用name_scope(...)之后，将压(push)新名称进栈中，
with tf.Graph().as_default() as g:
    c = tf.constant(5.0,name = 'c')
    print(c.op.name == 'c')
    c_1 = tf.constant(6.0,name = 'c')
    print(c_1.op.name == 'c_1')

c = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(c.get_shape())

