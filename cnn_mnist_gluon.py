import random
import sys
from io import BytesIO
import gzip
import struct
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet import gluon
import numpy as np
import cv2

def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return (label, image)


def Get_image_lable(img, lable):
    x = [random.randint(0, 9) for x in range(3)]
    black = np.zeros((28, 28), dtype='uint8')
    for i in range(3):
        if x[i] == 0:
            img[:, i * 28:(i + 1) * 28] = black
            lable[i] = 10
    return img, lable


class OCRIter():
    def __init__(self, count, batch_size, num_label, height, width, lable, image):
        self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.lable = lable
        self.image = image
        self.num_label = num_label

    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num = [random.randint(0, self.count - 1)
                       for i in range(self.num_label)]
                img, lab = Get_image_lable(np.hstack(
                    (self.image[x] for x in num)), np.array([self.lable[x] for x in num]))
                img = np.multiply(img, 1 / 255.0)
                data.append(img.reshape(1, self.height, self.width))
                label.append(lab)
                
            data_all = nd.array(data,ctx=mx.gpu())
            label_all = nd.array(label,ctx=mx.gpu())
            yield data_all,label_all

def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    
    for i in range(pred.shape[0] / 3):
        ok = True
        for j in range(3):
            k = i * 3 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
        if ok:
            hit += 1
    return hit


class Cont(nn.HybridBlock):
    def __init__(self,**kwargs):
        super(Cont,self).__init__(**kwargs)
        with self.name_scope():
            self.dese0 = nn.Dense(11)
            self.dese1 = nn.Dense(11)
            self.dese2 = nn.Dense(11)
    def hybrid_forward(self,F,X):
        return F.concat(*[self.dese0(X),self.dese1(X),self.dese2(X)],dim=0)

def GetNet():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=32,kernel_size=5,activation='relu'))
        net.add(nn.MaxPool2D(pool_size=2,strides=1))
        
        net.add(nn.Conv2D(channels=32,kernel_size=5,activation='relu'))
        net.add(nn.AvgPool2D(pool_size=2,strides=1))
        
        net.add(nn.Conv2D(channels=32,kernel_size=3,activation='relu'))
        net.add(nn.AvgPool2D(pool_size=2,strides=1))
        
        net.add(nn.Conv2D(channels=32,kernel_size=3,activation='relu'))
        net.add(nn.AvgPool2D(pool_size=2,strides=1))
        
        net.add(nn.Flatten())
        net.add(nn.Dense(256))
        net.add(Cont())
        return net

if __name__ == '__main__':

    net = GetNet()
    #net.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34),ctx=mx.gpu())
    net.load_params("cnn_mnist_gluon",ctx=mx.gpu())
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    #trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001,'wd': 0.0001,'momentum':0.9})
    trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': 0.001})

    (train_lable, train_image) = read_data(
        './mnist/train-labels-idx1-ubyte.gz', './mnist/train-images-idx3-ubyte.gz')
    (test_lable, test_image) = read_data(
        './mnist/t10k-labels-idx1-ubyte.gz', './mnist/t10k-images-idx3-ubyte.gz')

    batch_size = 64
    data_train = OCRIter(1280, batch_size, 3, 28,84, train_lable, train_image)
    data_test = OCRIter(5000, batch_size, 3, 28, 84, test_lable, test_image)

    from mxnet import autograd as autograd

    for epoch in range(500):
        train_loss = 0.
        train_acc = 0.
        for data,label in data_train:
            labell = nd.transpose(label)
            labell = nd.reshape(labell,(-1,))
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output,labell)
            loss.backward()
            trainer.step(batch_size)
            
            train_acc += Accuracy(label.asnumpy(),output.asnumpy())
            train_loss += nd.mean(loss).asscalar()
        print("Epoch %d. Loss: %f, Train acc %f" % (
                epoch, train_loss/1280, train_acc/1280))

    net.save_params("cnn_mnist_gluon")