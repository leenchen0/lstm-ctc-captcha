import tensorflow as tf
import numpy as np

from dataset import dataset
from model import get_model
from utils import decode_sparse_tensor, sparse_tuple_from, report_accuracy

DATA_FOLDER = './test'

# The width of the image output by CNN
MAX_TIMESTEPS = 58

def main():
    ds = dataset(DATA_FOLDER, 1)

    global_step = tf.Variable(0, trainable=False)

    outputs, inputs, _, seq_len = get_model()
    decoded, _ = tf.nn.ctc_beam_search_decoder(outputs, seq_len, merge_repeated=False)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        saver.restore(sess, 'trained_model/model')

        test_input, test_label = ds.data, ds.labels
        test_targets = sparse_tuple_from(test_label)

        feed = {inputs: test_input, seq_len: [MAX_TIMESTEPS for _ in range(len(test_input))]}
        dd = sess.run(decoded[0], feed_dict=feed)
        report_accuracy(dd, test_targets)

if __name__ == "__main__":
    main()
