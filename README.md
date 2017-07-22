# char-rnn-tensorflow
Multi-layer long short-term memory (LSTM) char-level neural network implemented in Tensorflow.

## Usage

For a basic demonstration on the Shakespeare dataset, run the demo.py script using `python demo.py`. This will generate text from pre-trained weights (25000 iterations), which, while marginally coherent, could use a lot more training. To train your own model, clone the repository, add any text file to the directory, and run `python train.py -i file_name`, where file_name is the relative path to the text file. Depending on the size and diversity of the text sample, you may want to change some parameters using command line arguments. You can see a list of these options using `python train.py --help`.

## Description

This is a Tensorflow implementation of a character level LSTM network, inspired by Andrej Karpathy's char-rnn, and partly adapted from Google's Deep Learning Udacity course. The model is initialized to standard hyperparameters like learning rate, batch size, number of hidden nodes, and number of layers, but these can all be controlled using various command line arguments.

The default settings may not be ideal for a given dataset, but at least for a demonstration, only the data location needs to be specified. A larger validation set may be desirable for a large dataset, and more layers and nodes will make the network more accurate, but will also slow training. In my experience, deeper networks will require a lower learning rate for faster convergence.

### Compatibility

The model has only been tested on Python 3.5, but it should work on Python 2.7 (possibly with some minor tweaks). I will try my best to add documentation and expand the model as time permits.
