import numpy as np
import scipy.io
import os
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

#从VGG模型中进行加载 要网络结构一样
def net(data_path,input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    #去取平均值
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean,axis=(0,1))
    weights = data['layer'][0]
    net = {}
    current = input_image
    for i,name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            kernels = np.transpose(current,kernels,bias)
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
        assert len(net) == len(layers)
        return net, mean_pixel, layers


print("Network for VGG ready")
#加载当前模型
cwd = os.getcwd()
VGG_PATH = cwd + "/data/imagenet-vgg-verydeep-19.mat"
IMG_PATH = cwd + "/data/cat.jpg"
input_image = imread(IMG_PATH)
shape = (1,input_image.shape[0],input_image.shape[1],input_image.shape[2])
with tf.Session() as sess:
    image = tf.placeholder('float',shape = shape)
    nets,mean_pixel,all_layers = net(VGG_PATH,image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    layers = all_layers  # For all layers
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')
    for i, layer in enumerate(layers):
        print("[%d/%d] %s" % (i + 1, len(layers), layer))
        features = nets[layer].eval(feed_dict={image: input_image_pre})

        print(" Type of 'features' is ", type(features))
        print(" Shape of 'features' is %s" % (features.shape,))
        # Plot response
        if 1:
            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i + 1)
            plt.title("" + layer)
            plt.colorbar()
            plt.show()