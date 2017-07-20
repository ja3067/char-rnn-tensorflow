# char-rnn-tensorflow
Multi-layer long short-term memory (LSTM) char-level neural network implemented in Tensorflow.
         
## Usage

This is a Tensorflow implementation of a character level LSTM network, inspired by Andrej Karpathy's char-rnn, and partly adapted from Google's Deep Learning Udacity course. The model is initialized to standard hyperparameters like learning rate, batch size, number of hidden nodes, and number of layers, but these can all be controlled using various command line arguments. To see a list of command line arguments, use the -h handle, e.g. python LSTM.py -h.

The default settings may not be ideal for a given dataset, but at least for a demonstration, only the data location needs to be specified. A larger validation set may be desirable for a large dataset, and more layers and nodes will make the network more accurate, but will also slow training. In my experience, deeper networks will require a lower learning rate for faster convergence. 

### Compatibility

The model has only been tested on Python 3.5, but it should work on Python 2.7 (possibly with some minor tweaks). I will try my best to add documentation and expand the model as time permits.
