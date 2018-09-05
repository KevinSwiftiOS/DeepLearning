import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

print("packs loaded")
print ("Download and Extract MNIST dataset")
mnist = input_data.read_data_sets("data/",one_hot = True)
print("type of mnist is %s" %(type(mnist)))
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))

training = mnist.train.images
traininglabel = mnist.train.labels
testing = mnist.test.images
testinglabel = mnist.test.labels

print (" type of 'trainimg' is %s"    % (type(training)))
print (" type of 'trainlabel' is %s"  % (type(traininglabel)))
print (" type of 'testimg' is %s"     % (type(testing)))
print (" type of 'testlabel' is %s"   % (type(testinglabel)))
print (" shape of 'trainimg' is %s"   % (training.shape,))
print (" shape of 'trainlabel' is %s" % (traininglabel.shape,))
print (" shape of 'testimg' is %s"    % (testing.shape,))
print (" shape of 'testlabel' is %s"  % (testinglabel.shape,))
nsample = 5
print(111)
print(training.shape[0])
print(222)
print(traininglabel[1,:])
#low 到 high 随机阐释几个数
randidx = np.random.randint(training.shape[0],size=nsample)
print(randidx)
for i in randidx:
    print(training[i,:])
    curr_img = np.reshape(training[i,:],(28 , 28))
    #返回沿轴方向的最大值得索引
    curr_label = np.argmax(traininglabel[i,:])
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i) + "th Training Data "
              + "Label is " + str(curr_label))
    print("" + str(i) + "th Training Data "
          + "Label is " + str(curr_label))
    plt.show()