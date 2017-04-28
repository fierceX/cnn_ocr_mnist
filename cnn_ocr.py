import logging
import random
import sys
from io import BytesIO

import mxnet as mx
import numpy as np
from captcha.image import ImageCaptcha

import cv2

sys.path.insert(0, "../../python")
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)


class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def Get_Image_label(nums,captcha,dim=(100,30)):
    images = []
    label = []
    for num in nums:
        img = captcha.generate(num)
        img = np.fromstring(img.getvalue(), dtype='uint8')
        img = cv2.imdecode(img,cv2.IMREAD_COLOR)
        img = cv2.resize(img,dim)
        img = np.multiply(img,1/255.0)
        img = img.transpose(2,0,1)
        images.append(img)
        label.append([int(x) for x in num])
    
    return (images,label)
        


class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['./Ubuntu-M.ttf'])

        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]
        
    def __iter__(self):
        for k in range(self.count / self.batch_size):

            nums = [[str(random.randint(0,9)) for z in range(4)] for x in range(batch_size)]
            (imgs,labels) = Get_Image_label(nums,self.captcha,(self.width,self.height))
            data_all = [mx.nd.array(imgs)]
            label_all = [mx.nd.array(labels)]
            data_names = ['data']
            label_names = ['softmax_label']
            
            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_ocrnet():
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
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 256)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 10)
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24], dim = 0)

    label = mx.symbol.transpose(data = label)
    label = mx.symbol.reshape(data = label, shape = (-1, ))

    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 4):
        ok = True
        for j in range(4):
            k = i * 4 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total

if __name__ == '__main__':
    network = get_ocrnet()
    devs = [mx.cpu(i) for i in range(1)]

    #_, arg_params, __ = mx.model.load_checkpoint("cnn-orc", 1)

    model = mx.mod.Module(network, context=devs)

    batch_size = 16 
    data_train = OCRIter(100000, batch_size, 4, 30, 100)
    data_test = OCRIter(1000, batch_size, 4, 30, 100)

    model.fit(
        data_train,
        eval_data=data_test,
        num_epoch=1,
        optimizer='sgd',
        eval_metric=Accuracy,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        optimizer_params={'learning_rate': 0.001, 'wd': 0.00001},
        batch_end_callback=mx.callback.Speedometer(batch_size, 50),
    )
    model.save_checkpoint(prefix="cnn-orc", epoch=2)