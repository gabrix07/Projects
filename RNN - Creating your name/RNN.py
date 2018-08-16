import numpy as np
import matplotlib.pyplot as plt


iteration = 100000
neuronen_hidden = [27,50] 
neuronen_out = [27]
learning_rate=0.01
names = 'personal_names.txt'


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



def rnn_cell_forward(X, a_prev, parameters):
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    a_next = np.tanh(np.dot(Wax, X) + np.dot(Waa, a_prev) + ba)
    Z = np.dot(Wya, a_next) + by
    yt_pred = softmax(Z)
    cache = (a_next, a_prev, X, parameters)
    return a_next, yt_pred, cache



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def rnn_forward(X, a0, parameters):
    n_x, m, T_x = X.shape
    n_y, n_a = parameters["Wya"].shape
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    caches = []
    for t in range(T_x):
        a_next, yt_pred, cache = rnn_cell_forward(X[:,:,t], a_next, parameters)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
        caches.append(cache)
    return a, y_pred, caches

def compute_cost(y_pred, Y):
    cost = 0
    for t in range(T_y):
        cost += np.sum(-Y[:,:,t]*np.log(y_pred[:,:,t]))
    return cost


def linear_backward(dZy, cache, da_prevt):
    gradients = {}
    a_next, a_prev, X_t, parameters = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    gradients['dWya'] = np.dot(dZy, a_next.T) 
    gradients['dby'] = np.sum(dZy, 1, keepdims=True)
    dA_z = np.dot(Wya.T, dZy)
    dZ_a = (1 - a_next**2) * (dA_z + da_prevt)
    dZ_x = np.dot(Wax.T, dZ_a)
    gradients['dWax'] = np.dot(dZ_a, X_t.T)
    da_prevt = np.dot(Waa.T, dZ_a)
    gradients['dWaa'] = np.dot(dZ_a, a_prev.T)
    gradients['dba'] = np.sum(dZ_a, 1, keepdims=True)
    #print(gradients)
    return gradients, da_prevt


def model_backward(y_pred, Y, caches):
    dZ = y_pred - Y
    da_prevt = np.zeros((50, 1))
    grads = []
    dWya =  np.zeros((27, 50))
    dby = np.zeros((27, 1))
    dWax = np.zeros((50, 27))
    dWaa = np.zeros((50, 50))
    dba = np.zeros((50, 1))
    for t in reversed(range(T_x)):
        gradients, da_prevt = linear_backward(dZ[:,:,t], caches[t], da_prevt)
        dWyat, dbyt, dWaxt, dWaat, dbat =  gradients["dWya"], gradients["dby"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dWya += dWyat
        dby += dbyt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
    grads = {"dWya": dWya, "dby": dby, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    return grads
    


def update_parameters(parameters,grads,learning_rate):
     L = len(parameters) # number of layers in the neural network
     for l in range(L):
        parameters[l]["Wya"] = parameters[l]["Wya"] - learning_rate * grads["dWya"]
        parameters[l]["by"] = parameters[l]["by"] - learning_rate * grads["dby"]
        parameters[l]["Waa"] = parameters[l]["Waa"] - learning_rate * grads["dWaa"]
        parameters[l]["Wax"] = parameters[l]["Wax"] - learning_rate * grads["dWax"]
        parameters[l]["ba"] = parameters[l]["ba"] - learning_rate * grads["dba"]        
     return parameters



def clip(gradients, maxValue):
    dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['dba'], gradients['dby']
    for gradient in [dWax, dWaa, dWya, dba, dby]:
        np.clip(gradient,-maxValue , maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "dba": dba, "dby": dby}
    return gradients


def sample(parameters, char_to_ix, seed):
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix['\n']
    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+b)
        z = np.dot(Wya,a)+by
        y = softmax(z)
        np.random.seed(counter + seed)
        idx = np.random.choice(range(len(y)),p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        a_prev = a
        seed += 1
        counter += 1
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    return indices



seed = 0
data, vocab_size, char_to_ix, ix_to_char, examples = create_set(names)
a_prev = np.zeros((neuronen_hidden[1], 1))
parameters = initialize_parameters(neuronen_hidden, neuronen_out)
for loop in range(iteration):
    index = loop % len(examples)
    X_list = [None] + [char_to_ix[ch] for ch in examples[index]] 
    Y_list = X_list[1:] + [char_to_ix["\n"]]
    T_x = len(X_list)
    T_y = len(Y_list)
    X_train, Y_train = list_to_vector(X_list, Y_list, T_x, T_y)
    a, y_pred, caches= rnn_forward(X_train, a_prev, parameters[0])
    cost = compute_cost(y_pred, Y_train)
    if loop % 3500 == 0:
         print (cost)
         indices = sample(parameters[0], char_to_ix, seed)
         print_sample = [ix_to_char[ch] for ch in indices] 
         print(print_sample)
    grads = model_backward(y_pred, Y_train, caches)
    grads = clip(grads,5)
    parameters = update_parameters(parameters,grads,learning_rate)
    