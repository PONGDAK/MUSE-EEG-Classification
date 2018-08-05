# Imports
import tensorflow as tf
from util.utility import *

# Hyperparameters
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

with graph.as_default():
    # (batch, 128, 4) --> (batch, 64, 8)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 8) --> (batch, 32, 16)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 32, 16) --> (batch, 16, 32)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

    # (batch, 16, 32) --> (batch, 8, 64)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_4, (-1, 8 * 144))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    logits = tf.layers.dense(flat, n_classes)

    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with graph.as_default():
    saver = tf.train.Saver()

test_acc = []

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    predict = []
    lab = []
    correct = []
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}

        batch_acc, predict, correct, lab = sess.run([accuracy, tf.argmax(logits, 1), correct_pred, tf.argmax(labels_, 1)], feed_dict=feed)
        test_acc.append(batch_acc)

    for i in range(0, len(predict)):
        if i % 1 == 0:
            print("predict:label = {}:{} => {}".format(predict[i] + 1, lab[i] + 1, correct[i]))
        i += 1

    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))