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
parser.add_argument("-seed", dest="seed", required=False, help="specify seed for generated data", metavar="DATA", default='hobbits.')
parser.add_argument("-length", dest="length", required=False, help="length of output (number of lines)", metavar="LEN", default=1000)

args = parser.parse_args()

filename = os.path.abspath(os.path.expanduser(args.data))
data_name = os.path.splitext(os.path.basename(filename))[0]
seed = args.seed

def read_config(file):
    with open(os.path.join('config', os.path.splitext(os.path.basename(file))[0] + '.json')) as f:
        return dict(json.load(f))

save_folder = "saved_models"

config = read_config(filename)

batch_size = config['batch_size']
valid_size = config['valid_size']
num_unrollings = config['num_unrollings']
num_nodes = config['num_nodes']
num_layers = config['num_layers']
save_path = config['save_path']

def read_data(filename):
    with open(filename, 'r') as f:
        data = tf.compat.as_str(f.read())
    return data

print("Loading data...")

pattern = re.compile('([^\s\w.,:/]|_)+') # used to sanitize inputs. change as desired.

text = pattern.sub('', read_data(filename))

char_list = sorted(list(set(text)))

print(char_list)

chars = dict(enumerate(char_list))
reverse_chars = dict(zip(chars.values(), chars.keys()))

print('Data size %d. Number of unique chars %d' % (len(text), len(chars)))

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

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    # with open(os.path.join(save_folder, 'checkpoint'), 'r') as f:
    #     line = f.readline()
    # save_path = os.path.join(save_folder, line.split(': ')[-1].rstrip().replace('"', ''))

    print('Graph initialized. Loading weights from {}'.format(save_path))

    saver.restore(session, save_path)
    print("Model restored.")

    print('=' * 80)
    reset()
    for c in seed:
        feed = (np.arange(len(chars)) == char2id(c)).astype(np.uint8).reshape(1, len(chars))
        feed = sample_prediction.eval({sample_input: feed})
    sentence = ""
    for _ in range(10000):
        prediction = sample_prediction.eval({sample_input: feed})
        feed = sample(prediction)
        sentence += characters(feed)[0] # return character based on output distribution (as opposed to using argmax)
    print(sentence)
    print('=' * 80)

    with open('sample.txt', 'w') as f:
        f.write(sentence)

    # Measure validation set perplexity.
    reset()
    valid_logprob = 0
    for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
    print('Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size)))