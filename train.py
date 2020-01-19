import tensorflow as tf
import numpy as np

from dataset import dataset
from model import get_model
from utils import decode_sparse_tensor, sparse_tuple_from, report_accuracy

DATA_FOLDER = './train'
BATCH_SIZE = 40
NUM_EPOCHES = 200
REPORT_EPOCHES = 5

INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9

# The width of the image output by CNN
MAX_TIMESTEPS = 58

def main():
    ds = dataset(DATA_FOLDER, BATCH_SIZE)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)

    outputs, inputs, targets, seq_len = get_model()

    loss = tf.nn.ctc_loss(labels=targets, inputs=outputs, sequence_length=seq_len)
    cost = tf.reduce_mean(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

    decoded, _ = tf.nn.ctc_beam_search_decoder(outputs, seq_len, merge_repeated=False)
    e_dis = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    init = tf.global_variables_initializer()

    def do_report():
        test_inputs, test_labels, _ = ds.next_batch()
        test_targets = sparse_tuple_from(test_labels)
        test_feed = {inputs: test_inputs, targets: test_targets, seq_len: [MAX_TIMESTEPS for _ in range(len(test_inputs))]}
        dd = session.run(decoded[0], test_feed)
        report_accuracy(dd, test_targets)

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        saver.restore(session, 'saved_models/model-2382')
        for curr_epoch in range(NUM_EPOCHES):
            print("Epoch.......", curr_epoch)
            train_cost = 0
            new_epoch = False
            train_size = 0
            while not new_epoch:
                train_inputs, train_labels, new_epoch = ds.next_batch()
                train_targets = sparse_tuple_from(train_labels)
                feed = {inputs: train_inputs, targets: train_targets, seq_len: [MAX_TIMESTEPS for _ in range(len(train_inputs))]}
                c, steps, _ = session.run([cost, global_step, optimizer], feed)

                train_cost += c * BATCH_SIZE
                print("Step: %d, Loss: %.5f" % (steps, c))

                train_size += BATCH_SIZE

            if (curr_epoch + 1) % REPORT_EPOCHES == 0:
                do_report()
                save_path = saver.save(session, "saved_models/model", global_step=steps)
                print('save model on %s' % save_path)

            train_cost /= train_size

            train_inputs, train_labels, _ = ds.next_batch()
            train_targets = sparse_tuple_from(train_labels)
            val_feed = {inputs: train_inputs, targets: train_targets, seq_len: [MAX_TIMESTEPS for _ in range(len(train_inputs))]}
            val_cost, val_edit_dis, lr, steps = session.run([cost, e_dis, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}, val_edit_dis = {:.3f}, learning_rate = {}"
            print(log.format(curr_epoch + 1, NUM_EPOCHES, steps, train_cost, val_cost, val_edit_dis, lr))

if __name__ == "__main__":
    main()
