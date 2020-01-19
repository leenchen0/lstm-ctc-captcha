import tensorflow as tf
import 

from captcha_generator import CAPTCHA_SIZE, CHARSET

NUM_CLASSES = len(CHARSET) + 1
LSTM_LAYERS = [128, 128]

class Model:
    
    __init__(self):
        self.

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.5)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1], strides=[1, stride[0], stride[1], 1], padding='SAME')

def lstm_cell(num_hidden):
    return tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

def conv_layer(inputs):
    # 36*230*3 => 18*115*48
    W_conv1 = weight_variable([5, 5, 3, 48])
    b_conv1 = bias_variable([48])
    h_conv1 = tf.nn.relu(conv2d(inputs, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # 18*115*48 => 9*115*64
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # 9*115*64 => 5*58*128
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return h_pool3

def lstm_layer(inputs, seq_len, layers):
    stack = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(num_hidden) for num_hidden in layers],
        state_is_tuple=True
    )
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    return outputs

def full_connected_layer(inputs, num_hidden, num_classes):
    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    outputs = tf.matmul(inputs, W) + b

    return outputs

def get_model():
    inputs = tf.placeholder(tf.float32, [None, CAPTCHA_SIZE[1], CAPTCHA_SIZE[0], 3])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])

    outputs = conv_layer(inputs)

    _, feature_h, feature_w, channels = outputs.get_shape().as_list()
    outputs = tf.transpose(outputs, [0, 2, 1, 3]) # batchsize*feature_w*feature_h*channels
    outputs = tf.reshape(outputs, [-1, feature_w, feature_h * channels]) # batchsize*feature_w*(feature_h*channels)

    outputs = lstm_layer(outputs, seq_len, LSTM_LAYERS)

    num_hidden = LSTM_LAYERS[len(LSTM_LAYERS) - 1]
    outputs = tf.reshape(outputs, [-1, num_hidden])
    outputs = full_connected_layer(outputs, num_hidden, NUM_CLASSES)

    shape = tf.shape(inputs)
    outputs = tf.reshape(outputs, [shape[0], -1, NUM_CLASSES])
    outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs, inputs, targets, seq_len
