#coding=utf-8
# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
import os
from io import BytesIO
from collections import namedtuple
from genplate import *
chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ]
def getimage(GenPlate,dim=(544,144)):
    plateStr = GenPlate.genPlateString(-1,-1)
    img = GenPlate.generate(plateStr)
    img = cv2.resize(img,dim)
    cv2.imwrite('img.jpg',img)
    return img

def getnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")
    
    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")

    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 120)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc25 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc26 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc27 = mx.symbol.FullyConnected(data = fc1, num_hidden = 65)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24,fc25,fc26,fc27], dim = 0)

    softmax = mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")

    out = mx.symbol.Group([softmax, conv1, conv2, conv3, conv4])
    return out

def predict(img):

    img = cv2.resize(img,(120,30))
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,1,2)
    img = img.reshape(1,3,30,120)
    batch_size = 1
    net = getnet()

    mod = mx.mod.Module(symbol=net, context=mx.cpu())
    mod.bind(data_shapes=[('data', (batch_size, 3, 30, 120))])
    mod.load_params("./plate/cnn-ocr-plate-0095.params")

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(img)]))
    out = mod.get_outputs()
    prob = out[0].asnumpy()

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
        if i == 0:
            result =  np.argmax(prob[i][0:31])
        if i == 1:
            result =  np.argmax(prob[i][41:65])+41
        if i >  1:
            result =  np.argmax(prob[i][31:65])+31

        line += chars[result]
    return line

def RandImg():
    genplate = GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
    img = getimage(genplate)
    return img

def GetPlatePredict():
    img = RandImg()
    line = predict(img)
    line = line.encode('utf-8')
    return line

if __name__ == '__main__':
    img = RandImg()
    cv2.imshow("img",img)
    print predict(img)
    cv2.waitKey(0)
