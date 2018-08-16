import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


epochs = 3500
m = 1
hidden_neurons = 50
num_input = 27
num_classes = 27
names = 'personal_names.txt'
learning_rate=0.01



def initialize_parameters(neuronen_hidden, neuronen_out):
    np.random.seed(3)
    parameters = []
    parameters_temp = {}
    L = len(neuronen_hidden)           
    for l in range(1, L):
        parameters_temp['Wax'] = np.random.randn(neuronen_hidden[l] , neuronen_hidden[l - 1]) * 0.01
        parameters_temp['Waa'] = np.random.randn(neuronen_hidden[l] , neuronen_hidden[l]) * 0.01
        parameters_temp['ba'] = np.zeros((neuronen_hidden[l], 1))
        if (l == L-1):
            parameters_temp['Wya'] = np.random.randn(neuronen_out[0] , neuronen_hidden[l]) * 0.01
            parameters_temp['by'] = np.zeros((neuronen_out[0], 1))
        parameters.append(parameters_temp)
    return parameters



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
    X = np.zeros ((27,1,len(X_list)))
    Y = np.zeros ((27,1,len(Y_list)))
    for t in range(1,T_x):
        X[X_list[t],0,t] = 1
    for t in range(T_y):
        Y[Y_list[t],0,t] = 1
    return X, Y



data, vocab_size, char_to_ix, ix_to_char, examples = create_set(names)
index = 1 % len(examples)
X_list = [None] + [char_to_ix[ch] for ch in examples[index]] 
Y_list = X_list[1:] + [char_to_ix["\n"]]
T_x = len(X_list)
T_y = len(Y_list)
X_train, Y_train = list_to_vector(X_list, Y_list, T_x, T_y)
X_train = X_train.reshape(m, num_input , T_x)
Y_train = Y_train.reshape(m, num_classes , T_y)

X = tf.placeholder(tf.float32, [m, num_input, None])
Y = tf.placeholder(tf.int32, [m, num_classes, T_y])
init_state_a = tf.placeholder(tf.float32, [m, hidden_neurons])
current_state_a_test = tf.placeholder(tf.float32, [m, hidden_neurons])
X_test = tf.placeholder(tf.float32, [m, num_input])

W = tf.Variable(np.random.rand(hidden_neurons + num_input, hidden_neurons), dtype=tf.float32)
b = tf.Variable(np.zeros((1,hidden_neurons)), dtype=tf.float32)
W2 = tf.Variable(np.random.rand(hidden_neurons, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)


#inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(Y, axis=2)


current_state_a = init_state_a
states_series = []
for t in range (T_x):
    current_input = tf.reshape(X[:,:,t], [m,num_input])
    input_and_state_concatenated = tf.concat( [current_input, current_state_a],1)  # Increasing number of columns
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state


#labels_series = []
#for t in range (T_y):
#   current_output = np.reshape(Y_train[:,:,t], (m,num_classes))
#   labels_series.append(current_output) 

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.softmax_cross_entropy_with_logits(logits=logi, labels=lab) for logi, lab in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(total_loss)



input_and_state_concatenated_test = tf.concat( [X_test, current_state_a_test],1)
next_state_test = tf.tanh(tf.matmul(input_and_state_concatenated_test, W) + b)
predictions_series_test = tf.nn.softmax(tf.matmul(next_state_test, W2) + b2 )


state_a = np.zeros((m, hidden_neurons))
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range (epochs):
        _, loss  = sess.run([train_step, total_loss], feed_dict={X:X_train, init_state_a: state_a, Y: Y_train})
        print (loss)
    
    #Z_Train  = sess.run(labels_series, feed_dict={Y: Y_train})
    
    
    
    state_a_test = np.zeros((m, hidden_neurons))
    X_t = np.zeros((m, num_input))
    counter = 0
    idx = None
    while (idx != 0 and counter != 10):
        predictions_test, state_a_test = sess.run([predictions_series_test, next_state_test], feed_dict={X_test:X_t, current_state_a_test: state_a_test})
        idx = np.random.choice(num_classes,p=predictions_test.ravel())
        print (idx)
        X_t = np.zeros((m, num_input))
        X_t[:,idx] = 1
        counter += 1
        
        
        
#losses = [tf.nn.softmax_cross_entropy_with_logits(logits = logi, labels= lab) for logi, lab in zip(logits_series, labels_series)]
#losses = tf.nn.softmax_cross_entropy_with_logits(logits=predictions_series, labels=labels_series)
#total_loss = tf.reduce_mean(losses)
#train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
