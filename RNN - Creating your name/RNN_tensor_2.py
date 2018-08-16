import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import time


epochs = 3500
m = 1
hidden_neurons = 50
num_input = 27
num_classes = 27
names = 'personal_names.txt'
learning_rate=0.01


def create_set(names):
    data = open(names, 'r').read()
    data = data.lower()
    chars = list(set(data))
    vocab_size = len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    with open("personal_names.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    return data, vocab_size, char_to_ix, ix_to_char, examples



def list_to_vector(X_list, Y_list, T_x, T_y):
    X = np.zeros ((27,len(X_list),))
    Y = np.zeros ((27,1,len(Y_list)))
    for t in range(1,T_x):
        X[X_list[t],t] = 1
    for t in range(T_y):
        Y[Y_list[t],0,t] = 1
    return X, Y


def RNN(x, weights, biases):

    # reshape to [1, n_input]
    #x = tf.reshape(x, [-1, T_x])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,T_x,2)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(hidden_neurons)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, np.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



data, vocab_size, char_to_ix, ix_to_char, examples = create_set(names)
index = 1 % len(examples)
X_list = [None] + [char_to_ix[ch] for ch in examples[index]] 
Y_list = X_list[1:] + [char_to_ix["\n"]]
T_x = len(X_list)
T_y = len(Y_list)
X_train, Y_train = list_to_vector(X_list, Y_list, T_x, T_y)
X = tf.placeholder(tf.float32, [num_input,None])
Y = tf.placeholder(tf.int32, [None, num_classes])

weights = {
    'out': tf.Variable(tf.random_normal([hidden_neurons, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

#x = tf.reshape(X, [-1, T_x])
x = tf.split(X,T_x,1,1)
rnn_cell = rnn.BasicLSTMCell(hidden_neurons)

# generate prediction
outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)



# 1-layer LSTM with n_hidden units.
