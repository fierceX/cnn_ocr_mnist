import mxnet as mx
import numpy as np
import cv2
import random
from io import BytesIO
from collections import namedtuple
from cnn_ocr_mnist import read_data
import random


def get_ocrnet():
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
    pool1 = mx.symbol.Pooling(
        data=conv1, pool_type="max", kernel=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), num_filter=32)
    pool2 = mx.symbol.Pooling(
        data=conv2, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), num_filter=32)
    pool3 = mx.symbol.Pooling(
        data=conv3, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), num_filter=32)
    pool4 = mx.symbol.Pooling(
        data=conv4, pool_type="avg", kernel=(2, 2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data=relu4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=256)
    fc21 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc22 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc23 = mx.symbol.FullyConnected(data=fc1, num_hidden=10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23], dim=0)
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name="softmax")

    out = mx.symbol.Group([softmax, conv1, conv2, conv3, conv4])

    return out


def SetImage():
    (lable, image) = read_data(
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

    num = [random.randint(0, 5000 - 1)
           for i in range(3)]

    img = np.hstack((image[x] for x in num))

    imgw = 255 - img
    cv2.imwrite("img.jpg", imgw)
    img = np.multiply(img, 1 / 255.0)
    img = img.reshape(1, 1, 28, 84)
    return img


def ReadImage():
    img = cv2.imread('img.jpg', 0)
    #img = 255-img;
    img = cv2.resize(img, (84, 28))
    cv2.imwrite("aa.jpg", img)
    img = np.multiply(img, 1 / 255.0)
    img = img.reshape(1, 1, 28, 84)
    return img


def predict(img):
    _, arg_params, aux_params = mx.model.load_checkpoint("cnn-ocr-mnist", 1)
    net = get_ocrnet()

    mod = mx.mod.Module(symbol=net, context=mx.cpu())
    mod.bind(data_shapes=[('data', (1, 1, 28, 84))])
    mod.set_params(arg_params, aux_params)

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(img)]))
    out = mod.get_outputs()
    prob = out[0].asnumpy()
    pool4out = out[1].asnumpy()

    for n in range(4):
        cnnout = out[n + 1].asnumpy()
        width = int(np.shape(cnnout[0])[1])
        height = int(np.shape(cnnout[0])[2])
        cimg = np.zeros((width * 8 + 80, height * 4 + 40), dtype=float)
        cimg = cimg + 255
        k = 0
        for i in range(4):
            for j in range(8):
                cg = cnnout[0][k]
                cg = cg.reshape(width, height)
                cg = np.multiply(cg, 255)
                k = k + 1
                gm = np.ones((width + 10, height + 10), dtype=float)
                gm = gm + 255
                gm[0:width, 0:height] = cg
                cimg[j * (width + 10):(j + 1) * (width + 10), i *
                     (height + 10):(i + 1) * (height + 10)] = gm
        cv2.imwrite("c" + str(n) + ".jpg", cimg)

    line = ''
    for i in range(prob.shape[0]):
        line += str(np.argmax(prob[i]))
    return line


if __name__ == '__main__':
    img = SetImage()
    line = predict(img)
    print 'predicted: ' + line
