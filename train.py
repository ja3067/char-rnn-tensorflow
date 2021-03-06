from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
from six.moves import range
import re
import argparse
import json

def is_valid_type(arg):
    if not arg.endswith('.txt'):
        parser.error("%s is not a valid text file!" % arg)
    else:
        return arg

parser = argparse.ArgumentParser(description="Simple, versatile LSTM model.")
parser.add_argument("-i", dest="data", required=False, help="specify path to training data", metavar="DATA", type=is_valid_type, default='shakespeare.txt')
parser.add_argument("-valid_size", dest="valid_size", required=False, help="size of validation set", metavar="VALID", type=int, default=1000)
parser.add_argument("-batch_size", dest="batch_size", required=False, help="batch size", metavar="BATCH", type=int, default=50)
parser.add_argument("-num_unrollings", dest="num_unrollings", required=False, help="number of LSTM unrollings", metavar="UNROLLINGS", type=int, default=50) # 20
parser.add_argument("-num_nodes", dest="num_nodes", required=False, help="number of hidden nodes in each LSTM layer", metavar="NODES", type=int, default=128) # 512
parser.add_argument("-num_layers", dest="num_layers", required=False, help="number of hidden layers in LSTM", metavar="LAYERS", type=int, default=2)
parser.add_argument("-num_steps", dest="num_steps", required=False, help="number of iterations", metavar="ITER", type=int, default=50001)
parser.add_argument("-summary_frequency", dest="summary_frequency", required=False, help="how often to report loss", metavar="FREQ", type=int, default=500)
parser.add_argument("-init_rate", dest="init_rate", required=False, help="initial learning rate", metavar="INIT", type=float, default=.01) # 0.001
parser.add_argument("-decay_rate", dest="decay_rate", required=False, help="exponential decay rate", metavar="RATE", type=float, default=0.95)
parser.add_argument("-decay_steps", dest="decay_steps", required=False, help="number of iterations for each decay epoch", metavar="STEPS", type=float, default=5000)

args = parser.parse_args()

def save_config(namespace):
    config = vars(namespace)
    config['save_path'] = ""
    if not os.path.isdir('config'):
        os.mkdir('config')
    config_path = 'config/{}.json'.format(os.path.splitext(os.path.basename(namespace.data))[0])
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config_path


config_path = save_config(args)

filename = os.path.abspath(os.path.expanduser(args.data))

batch_size = args.batch_size
valid_size = args.valid_size

num_unrollings = args.num_unrollings
num_nodes = args.num_nodes
num_layers = args.num_layers

num_steps = args.num_steps
summary_frequency = args.summary_frequency

init_rate = args.init_rate
decay_rate = args.decay_rate
decay_steps = args.decay_steps # increase for larger models, lower learning rates, longer train times

valid_num_lines = 5 # increase to change number of sample lines displayed

def read_data(filename):
    with open(filename, 'r') as f:
        data = tf.compat.as_str(f.read())
    return data

print("Loading data...")

pattern = re.compile('([^\s\w.,:/]|_)+') # used to sanitize inputs. change as desired.

text = pattern.sub('', read_data(filename))

char_list = sorted(list(set(text)))
chars = dict(enumerate(char_list))
reverse_chars = dict(zip(chars.values(), chars.keys()))

print('Data size %d. Number of unique chars %d.' % (len(text), len(chars)))

try:
    assert valid_size < len(text)
except AssertionError:
    print("Size of validation set is larger than size of dataset. Please choose a smaller value or provide a large dataset.")

valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)

# Utility functions to map characters to vocabulary IDs and back.

vocabulary_size = len(chars)

def char2id(char):
    try:
        return reverse_chars[char]
    except KeyError:
        print('Unexpected character: %s' % char, ord(char))
        return -1


def id2char(dictid):
    try:
       return chars[dictid]
    except KeyError:
        print('Unexpected id %d' % dictid)
        return ' '

# Function to generate a training batch for the LSTM model.

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]


# Simple LSTM Model.
print("Initializing graph...")
graph = tf.Graph()
with graph.as_default():

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        # Input gate: input, previous output, and bias.
        x = tf.get_variable("x", [vocabulary_size, 4 * num_nodes],
                             initializer=tf.truncated_normal_initializer(-0.1, 0.1, dtype=tf.float32))
        m = tf.get_variable("im", [num_nodes, 4 * num_nodes],
                             initializer=tf.truncated_normal_initializer(-0.1, 0.1, dtype=tf.float32))

        ib = tf.get_variable("ib", [1, num_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))
        fb = tf.get_variable("fb", [1, num_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))
        cb = tf.get_variable("cb", [1, num_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))
        ob = tf.get_variable("ob", [1, num_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))

        xtemp = tf.matmul(i, x)
        mtemp = tf.matmul(o, m)

        ix, fx, cx, ox = tf.split(xtemp, 4, axis=1)
        im, fm, cm, om = tf.split(mtemp, 4, axis=1)

        input_gate = tf.sigmoid(ix + im + ib)
        forget_gate = tf.sigmoid(fx + fm + fb)
        update = cx + cm + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(ox + om + ob)

        return output_gate * tf.tanh(state), state

    # Input data.

    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    for j in range(num_layers):
        with tf.variable_scope('layer{}'.format(j)) as scope:
            w = tf.get_variable("w", [num_nodes, vocabulary_size],
                                initializer=tf.truncated_normal_initializer(-0.1, 0.1, dtype=tf.float32))
            b = tf.get_variable("b", [vocabulary_size], initializer=tf.zeros_initializer(dtype=tf.float32))

            saved_output = tf.get_variable("saved_output", [batch_size, num_nodes],
                                           initializer=tf.zeros_initializer(dtype=tf.float32), trainable=False)
            saved_state = tf.get_variable("saved_state", [batch_size, num_nodes],
                                          initializer=tf.zeros_initializer(dtype=tf.float32), trainable=False)

            # Unrolled LSTM loop.
            outputs = list()
            output = saved_output
            state = saved_state

            for i in inputs:
                output, state = lstm_cell(i, output, state)
                scope.reuse_variables()
                outputs.append(output)

            # State saving across unrollings.
            with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
                # Classifier.
                logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)

                if j != num_layers - 1:
                    _logits = tf.reshape(logits, [num_unrollings, batch_size, vocabulary_size])
                    inputs = tf.unstack(tf.nn.softmax(_logits))

                if j == num_layers - 1:
                    loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            labels=tf.concat(train_labels, 0), logits=logits))

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])


    def reset():
        for k in range(num_layers):
            with tf.variable_scope('layer{}'.format(k)) as scope:
                scope.reuse_variables()

                saved_sample_output = tf.get_variable("saved_sample_output", [1, num_nodes],
                                                      initializer=tf.zeros_initializer(dtype=tf.float32),
                                                      trainable=False)
                saved_sample_state = tf.get_variable("saved_sample_state", [1, num_nodes],
                                                     initializer=tf.zeros_initializer(dtype=tf.float32),
                                                     trainable=False)

                saved_sample_output.assign(tf.zeros([1, num_nodes]))
                saved_sample_state.assign(tf.zeros([1, num_nodes]))


    for j in range(num_layers):
        with tf.variable_scope('layer{}'.format(j)) as scope:

            saved_sample_output = tf.get_variable("saved_sample_output", [1, num_nodes],
                                                  initializer=tf.zeros_initializer(dtype=tf.float32), trainable=False)
            saved_sample_state = tf.get_variable("saved_sample_state", [1, num_nodes],
                                                 initializer=tf.zeros_initializer(dtype=tf.float32), trainable=False)

            scope.reuse_variables()

            w = tf.get_variable("w", [num_nodes, vocabulary_size],
                                initializer=tf.truncated_normal_initializer(-0.1, 0.1, dtype=tf.float32))
            b = tf.get_variable("b", [vocabulary_size], initializer=tf.zeros_initializer(dtype=tf.float32))

            if j == 0:
                sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)

            else:
                sample_output, sample_state = lstm_cell(temp_input, saved_sample_output, saved_sample_state)

            with tf.control_dependencies(
                    [saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):

                temp_input = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

                if j == num_layers - 1:
                    sample_prediction = temp_input

    saver = tf.train.Saver()

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(init_rate, global_step, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate) # RMSPropOptimizer, GradientDescentOptimizer
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Graph initialized.')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(valid_num_lines): # 5 by default
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0] # generate random character
                    reset()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0] # return character based on output distribution (as opposed to using argmax)
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))

            if step % 25000 == 0 or step == num_steps - 1:
                save_path = saver.save(session, "saved_models/{}model{}.ckpt".format(os.path.splitext(os.path.basename(filename))[0], step))
                print("Model saved in file: %s" % save_path)
                with open(config_path, 'r') as f:
                    config = dict(json.load(f))
                    config['save_path'] = save_path
                with open(config_path, 'w') as f:
                    json.dump(config, f, ensure_ascii=False)

