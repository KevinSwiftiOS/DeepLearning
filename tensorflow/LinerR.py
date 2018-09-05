import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#随机生成1000个点 分本在 y = 0.1x + 0.3 的直线周围
num_points = 1000
vectors_sets = []
x_data = []
y_data = []
for i in range(num_points):
    #均值为0 方差为0.55
    x1 = np.random.normal(0.0,0.55)
    #稍带有点误差
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0,0.03)
    x_data.append(x1)
    y_data.append(y1)
    vectors_sets.append([x1,y1])

#生成随机样本
# x_data = [v[0] for v in vectors_sets]
# y_data = [v[1] for v in vectors_sets]
#进行绘图
# plt.scatter(x_data,y_data,c = 'r')
# plt.show()


#进行预测 用tensotflow
#生成1维的W矩阵 取值是[-1,1]之间的随机数
W = tf.Variable(tf.random_uniform([1],-1.0,1.0),name = 'W')
#生成1维的b矩阵 初始值维0
b = tf.Variable(tf.zeros([1]), name='b')
#经过累计计算得出预估值y
y = W *x_data + b
#定义loss 前面是平均值
loss = tf.reduce_mean(tf.square(y - y_data),name = 'loss')
#定义优化器 梯度下降法
optimizer = tf.train.GradientDescentOptimizer(0.5)

#最小化loss
train = optimizer.minimize(loss,name = 'train')
#在session中进行操作
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print("W = ",sess.run(W),"b = ",sess.run(b),"loss = ",sess.run(loss))

for i in range(20):
        sess.run(train)
        print("W = ", sess.run(W), "b = ", sess.run(b), "loss = ", sess.run(loss))

#进行绘画

plt.scatter(x_data,y_data,c = 'r')
plt.plot(x_data,sess.run(W) * x_data + sess.run(b))
plt.show()