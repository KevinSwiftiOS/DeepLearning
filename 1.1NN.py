import  numpy as np
#定义激活函数 默认为False 为前向传播
def sigmoid(x,deriv =False):
      if(deriv == True):
          #说明为反向传播 返回求导值
          return x * (1 - x)
      return 1 / (1 + np.exp(-x))
#定义样本 为5个样本 每个样本3个特征
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1],
              [0,0,1]])
#定义样本值 为5 * 1 5个样本 1个参数值
y = np.array([[0],
              [1],
              [1],
              [1],
              [0]])
#随机初始化值 喂种子 每次随机值一样
np.random.seed(1)
#l0与l1之间的链接 有4个神经元 前面有3个特征 所以为3 * 4 要在-1 ~ 1 之间
w0 = 2 * np.random.random((3,4)) - 1
#当成输出层 输出惟一值
w1 = 2 * np.random.random((4,1)) - 1
for j in range(6000000):
    #l0为输入层
   l0 = x
    #矩阵的乘法
    #为前向传播
   l1 = sigmoid(np.dot(l0 , w0))
   l2 = sigmoid(np.dot(l1 , w1))
    #对比当前值与真实值的差异 损失函数为1/2(y - l2)的平方 需求导
   l2_error = y - l2
   #进行显示
   if(j % 10000 == 0):
        print("error " + str(np.mean(np.abs(l2_error))))
    #反向传播的求导操作 对应相乘 * 为 5 * 1 看w1做的贡献
   l2_delta = l2_error * sigmoid(l2,deriv=True)
    #w1为 4 * 1 l2的计算 对l1的影响
   l1_error = l2_delta.dot(w1.T)
   l1_delta = l1_error * sigmoid(l1,deriv=True)
   #更新值 上一层传下来的误差 自身的梯度也要计算出来
   w1 += l1.T.dot(l2_delta)
   w0 += l0.T.dot(l1_delta)







