import mxnet as mx
import numpy as np
import cv2
import random
from io import BytesIO
from collections import namedtuple
import random
from captcha.image import ImageCaptcha

def get_ocrnet():
    data = mx.symbol.Variable('data')

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

    return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")

def predict(img):
    _, arg_params, aux_params = mx.model.load_checkpoint("cnn-orc", 2)
    net = get_ocrnet()

    mod = mx.mod.Module(symbol=net, context=mx.cpu())
    mod.bind(data_shapes=[('data', (1, 1, 30, 100))])
    mod.set_params(arg_params, aux_params)

    Batch = namedtuple('Batch', ['data'])
    mod.forward(Batch([mx.nd.array(img)]))
    out = mod.get_outputs()
    prob = out[0].asnumpy()

    line = ''
    for i in range(prob.shape[0]):
        line += str(np.argmax(prob[i])
                    if int(np.argmax(prob[i])) != 10 else ' ')
    return line

def GetImage(captcha,num_label,shape=(100,30)):
    num = [str(random.randint(0,9)) for x in range(num_label)]
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    img = cv2.resize(img,shape)
    img = np.multiply(img,1/255.0)
    img = img.transpose(2,0,1)

    return img


if __name__ == '__main__':
    captcha = ImageCaptcha(fonts=['./Ubuntu-M.ttf'])
    img = GetImage(captcha,4)
    line = predict(img)
    print 'predicted: ' + line