import mxnet as mx
import numpy as np
import cv2
import random
from io import BytesIO
from collections import namedtuple
from cnn_mnist_gluon import read_data, Get_image_lable,GetNet
import random

def SetImage():
    (lable, image) = read_data(
        './mnist/t10k-labels-idx1-ubyte.gz', './mnist/t10k-images-idx3-ubyte.gz')
    num = [random.randint(0, 5000 - 1)
           for i in range(3)]

    img, _ = Get_image_lable(
        np.hstack((image[x] for x in num)), np.array([lable[x] for x in num]))
    imgw = 255 - img
    cv2.imwrite("img.jpg", imgw)
    img = np.multiply(img, 1 / 255.0)
    img = img.reshape(1, 1, 28, 84)
    return img,imgw

def predict(img,net):
    out = net(img)
    prob = out.asnumpy()
    line = ''
    for i in range(prob.shape[0]):
        line += str(np.argmax(prob[i])
                    if int(np.argmax(prob[i])) != 10 else ' ')
    
    return line
if __name__ == '__main__':
    net = GetNet()

    net.load_params("cnn_mnist_gluon",ctx=mx.cpu())
    net.hybridize()
    img,imgs = SetImage()
    #cv2.imshow("img",imgs)
    line = predict(mx.nd.array(img,ctx=mx.cpu()),net)
    print 'predicted: ' + line
    #cv2.waitKey(0)