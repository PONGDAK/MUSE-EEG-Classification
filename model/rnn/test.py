# Imports
import tensorflow as tf
from util.utility import *

# Hyperparameters
lstm_size = 12  # 3 times the amount of channels
lstm_layers = 3  # Number of layers
batch_size = 720  # Batch size.
seq_len = 128  # Number of steps
learning_rate = 0.01  # Learning rate (default is 0.001)
epochs = 1000

# Fixed
n_classes = 6
n_channels = 4

# Prepare data
print('------    load data    ------')

X_test, labels_test = load_data('./../../filter/test_data.csv', './../../filter/label_test.csv', seq_len)

# Standardize
X_test = standardize(X_test)

# One-hot encoding:
y_test = one_hot(labels_test)

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name='inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

# Construct inputs to LSTM
with graph.as_default():
    # Construct the LSTM inputs and LSTM cells
    lstm_in = tf.transpose(inputs_, [1, 0, 2])  # reshape into (seq_len, N, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_channels])  # Now (seq_len*N, n_channels)

    # To cells
    lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, seq_len, 0)

    # Add LSTM layers
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

# Define forward pass, cost function and optimizer:
with graph.as_default():
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state=initial_state)

    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_))
    # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

    # Grad clipping
    train_op = tf.train.AdamOptimizer(learning_rate_)

    gradients = train_op.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    optimizer = train_op.apply_gradients(capped_gradients)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with graph.as_default():
    saver = tf.train.Saver()

# Evaluate on train set
test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))

    predict = []
    lab = []
    correct = []
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1,
                initial_state: test_state}

        batch_acc, test_state, predict, correct, lab = sess.run(
            [accuracy, final_state, tf.argmax(logits, 1), correct_pred, tf.argmax(labels_, 1)], feed_dict=feed)

        test_acc.append(batch_acc)

    for i in range(0, len(predict)):
        if i % 1 == 0:
            print("predict:label = {}:{} => {}".format(predict[i] + 1, lab[i] + 1, correct[i]))
        i += 1

    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
