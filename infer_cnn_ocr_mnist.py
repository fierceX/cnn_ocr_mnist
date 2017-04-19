import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random
from io import BytesIO
from cnn_ocr_mnist import gen_sample , read_data, gen_saplei


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
    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23], dim = 0)
    return mx.symbol.SoftmaxOutput(data = fc2, name = "softmax")

def predict():
    (lable,image) = read_data('train-labels-idx1-ubyte.gz','train-images-idx3-ubyte.gz')

    (num,img) = gen_saplei(60000,image)

    imgw = 255 - img
    cv2.imwrite("img.jpg",imgw)

    img = np.multiply(img, 1 / 255.0)
    img = img.reshape(1,28,84)

    
    
    batch_size = 1
    _, arg_params, __ = mx.model.load_checkpoint("cnn-ocr-mnist", 1)
    data_shape = [("data", (batch_size, 1, 28, 84))]
    input_shapes = dict(data_shape)
    sym = get_ocrnet()

    executor = sym.simple_bind(ctx = mx.cpu(), **input_shapes)

    for key in executor.arg_dict.keys():
        if key in arg_params:
            arg_params[key].copyto(executor.arg_dict[key])

    executor.forward(is_train = True, data = mx.nd.array([img]))
    probs = executor.outputs[0].asnumpy()
    line = ''
    for i in range(probs.shape[0]):
        line += str(np.argmax(probs[i]))
    return line

if __name__ == '__main__':
    
    line = predict()
    print 'predicted: ' + line
