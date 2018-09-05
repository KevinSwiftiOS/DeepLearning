#保存与加载的操作
import tensorflow as tf
v1 = tf.Variable(tf.random_normal([1,2]),name = 'v1')
v2 = tf.Variable(tf.random_normal([2,3]),name = "v2")
#初始化
init_op = tf.global_variables_initializer()
#进行保存
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     print("V1:",sess.run(v1))
#     print("V2:",sess.run(v2))
#     #保存到当前目录下 将整个会话保存下来
#     saver_path = saver.save(sess,"save/model.ckpt")
#     print("Model saved in file:",saver_path)

#进行读取 虽然v1和v2是随机初始化的 但是读取出来是一样的
saver = tf.train.Saver()
with tf.Session() as sess:
    #读取
    saver.restore(sess,"save/model.ckpt")
    print("V1:", sess.run(v1))
    print("V2:", sess.run(v2))
    print(("model restored"))