import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
#import time
from tqdm import tqdm
import tensorflow.contrib.layers as layers

#from cnn_utils import *


epoch = 31
batch = 5000
mini_batch = 250
test_percentage = 0.1
classes = 2
cnn_dims = [11,5,3,3,3]
cnn_channels = [3,96,256,384,384,256]
cnn_stride = [4,1,1,1,1]
cnn_padding = ["VALID","SAME","SAME","SAME","SAME"]
pool_dim = [3,3,0,0,3]
pool_stride = [2,2,0,0,2]
pool_padding = ["VALID","VALID","VALID","VALID","VALID"]
fc_neurons = [4096,4096] 
weight_decay = 0.1
learning_rate=0.001
image_size = np.array([227,227])

log_string = 'cd={},cs={},cc={},pd={},pl={},fc={},lr={}'.format(cnn_dims,
                 cnn_channels, cnn_stride, pool_dim, pool_stride, fc_neurons, learning_rate)
log_string = log_string.replace(" ", "")


def Create_a_dataset(classes, batch , image_size, test_percentage):
    images = []
    test_size = round(batch*test_percentage)
    number_image = test_size + batch
    Y =[]
    n_class, res = divmod(number_image,classes)
    for i in range (classes):
        filename = 0
        data_classes = n_class
        while filename < data_classes:
            img = cv2.imread('./Images/' + str(i) + '/'  + str(filename) + '.jpg')
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size[0], image_size[1])) 
                filename = filename + 1
                images.append(img)  
                Y.append(i)
            else:
                data_classes = data_classes + 1 
                filename = filename + 1 
        while res > 0:
            img = cv2.imread('./Images/' + str(i) + '/'  + str(filename) + '.jpg')
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size[0], image_size[1])) 
                res = res - 1
                images.append(img)  
                Y.append(i)
            else:
                filename = filename + 1    
    X_train, X_test, y_train, y_test = train_test_split(images, Y,
                                                        test_size = test_size,
                                                        random_state = 2)
    return X_train, X_test, y_train, y_test


def create_vector(x_train, x_test, y_train, y_test, classes):
    l_train = len(x_train)
    l_test = len(x_test)    
    X_train = np.zeros((l_train, image_size[0], image_size[1], 3), dtype=np.uint8)
    X_test = np.zeros((l_test, image_size[0], image_size[1], 3), dtype=np.uint8)
    Y_train = np.zeros((l_train, classes))
    Y_test = np.zeros((l_test, classes))
    for i in range(l_train):
        X_train[i,:,:,:] = x_train[i]
        Y_train[i, y_train[i]] = 1
    for i in range(l_test):
        X_test[i,:,:,:] = x_test[i]
        Y_test[i, y_test[i]] = 1
    return X_train, X_test, Y_train, Y_test


def mini_batch_image(X_train,Y_train, batch, mini_batch) :
    num_minibatches = int(batch / mini_batch) 
    X_train_mini = []
    Y_train_mini = []
    X_train_m = np.zeros((mini_batch, X_train.shape[1], X_train.shape[2], 3), dtype=np.uint8)   
    Y_train_m = np.zeros((mini_batch, Y_train.shape[1]))
    for i in range (num_minibatches):
        X_train_m = X_train[i * mini_batch : mini_batch*(1 + i),:,:,:]
        Y_train_m = Y_train[i * mini_batch : mini_batch*(1 + i),:]
        Y_train_mini.append(Y_train_m)
        X_train_mini.append(X_train_m) 
    return X_train_mini, Y_train_mini, num_minibatches

        
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "Inputs")
    Y = tf.placeholder(tf.float32, [None, n_y], name = "Labels")
    return X, Y


def initialize_parameters(cnn_dims, cnn_channels):
    tf.set_random_seed(1)                             
    L = len(cnn_channels)
    parameters= {}
    for l in range(1, L):
        #parameters['W' + str(l)] = tf.get_variable("W" + str(l), [cnn_dims[l-1], cnn_dims[l-1], cnn_channels[l-1], cnn_channels[l]], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        parameters['W' + str(l)] = tf.Variable(tf.truncated_normal([cnn_dims[l-1],
                   cnn_dims[l-1], cnn_channels[l-1],
                   cnn_channels[l]], stddev=0.1), name="W" + str(l))
    return parameters


def Conv_Layer(X, parameters, pool_dim, pool_stride, cnn_stride, cnn_padding): 
    L = len(parameters)  
    P = X
    for l in range(L): 
        tf.summary.histogram("Weight_CL-"+ str(l+1), parameters['W' + str(l+1)])
        #W = tf.Variable(tf.truncated_normal([cnn_dims[l], cnn_dims[l], cnn_channels[l], cnn_channels[l+1]], stddev=0.1), name="W" + str(l))
        Z = tf.nn.conv2d(P,parameters['W' + str(l+1)] , strides=[1, cnn_stride[l],
                                      cnn_stride[l], 1], padding= cnn_padding[l])
        P = tf.nn.relu(Z)
        if pool_dim[l] != 0:
            P = tf.nn.max_pool(P, ksize = [1, pool_dim[l], pool_dim[l], 1], 
                               strides = [1, pool_stride[l], pool_stride[l], 1],
                               padding=pool_padding[l])
    P = tf.contrib.layers.flatten(P) # FLATTEN
    return P


def Fully_connected(P, fc_neurons):
    L_conec = len(fc_neurons)
    for l in range(0, L_conec):       
        P = tf.contrib.layers.fully_connected(P, 
                                              fc_neurons[l],
                                              scope = 'FC-'+str(l),
                                              weights_regularizer = layers.l2_regularizer(weight_decay),
                                              biases_regularizer=layers.l2_regularizer(weight_decay))
    Z_out = tf.contrib.layers.fully_connected(P, 2, activation_fn=None,
                                              scope = 'FC-'+str(l+1),
                                              weights_regularizer=layers.l2_regularizer(weight_decay),
                                              biases_regularizer=layers.l2_regularizer(weight_decay))
    return Z_out, P


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def create_sprite_image_and_labels(images, image_size, y_test):
    n_plots = int(np.ceil(np.sqrt(len(images))))
    spriteimage = np.zeros((image_size[0] * n_plots ,image_size[1] * n_plots, 3 ), dtype=np.uint8)
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < len(images):
                this_img = images[this_filter]
                spriteimage[i * image_size[0]:(i + 1) * image_size[0],
                            j * image_size[1]:(j + 1) * image_size[1],:] = this_img
    plt.imsave('./test/' + log_string + '/Test.png', spriteimage)    
    with open('./test/' + log_string + '/Test.tsv','w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(y_test):
            f.write("%d\t%d\n" % (index,label))
    return 


def embedding_visualization(y_test, x_test, fc_neurons, embedding_input, image_size, test_writer):
    create_sprite_image_and_labels(x_test, image_size, y_test)        
    embedding = tf.Variable(tf.zeros([len(y_test), fc_neurons[0]]), name='Test_embedding')
    assignment = embedding.assign(embedding_input)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.single_image_dim.extend(image_size)
    embedding_config.sprite.image_path = 'Test.png'
    embedding_config.metadata_path =  'Test.tsv'
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(test_writer, config)
    return assignment


def get_var(name,all_vars):
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None






# In[0]:
x_train, x_test, y_train, y_test = Create_a_dataset(classes, batch , image_size, test_percentage)
X_train, X_test, Y_train, Y_test = create_vector(x_train, x_test, y_train, y_test, classes) 
X_train_mini, Y_train_mini, num_minibatches = mini_batch_image(X_train,Y_train, batch, mini_batch)
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(image_size[0], image_size[1], cnn_channels[0], 2)
    with tf.name_scope("Conv_Layer"):        
        parameters = initialize_parameters(cnn_dims, cnn_channels)
        P = Conv_Layer(X, parameters, pool_dim, pool_stride, cnn_stride, cnn_padding)
    Z_out, embedding_input = Fully_connected(P, fc_neurons)
    with tf.name_scope('cost'):
        cost = compute_cost(Z_out, Y)
        cost_sum = tf.summary.scalar('cost', cost)
    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    with tf.name_scope("Accuracy"):
        predictions = tf.nn.softmax(Z_out)
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred[:,1], tf.float32))
        accuracy_sum = tf.summary.scalar('Accuracy', accuracy)
    
    tf.summary.image('Input', X, 3)
    all_vars = tf.global_variables()
    for l in range(len(fc_neurons) + 1):
        tf.summary.histogram("Biases_FC-"+ str(l), get_var('FC-' + str(l) + '/biases',all_vars))
        tf.summary.histogram("Weight_FC-"+ str(l), get_var('FC-' + str(l) + '/weight',all_vars))
    
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    merged_summary_test = tf.summary.merge([cost_sum, accuracy_sum])
    train_writer = tf.summary.FileWriter('./train/' + log_string)
    train_writer.add_graph(sess.graph)
    test_writer = tf.summary.FileWriter('./test/' + log_string)
    assignment = embedding_visualization(y_test, x_test, fc_neurons,
                                         embedding_input, image_size,
                                         test_writer)
    saver = tf.train.Saver()
    for e in range (epoch):
        #tic=time.time()
        #with tqdm(total= num_minibatches*mini_batch) as pbar:
        for i in tqdm(range (num_minibatches),desc = 'Epoch ' + str(e)):           
            sess.run(optimizer, feed_dict={X:X_train_mini[i], Y:Y_train_mini[i]}) 
                #pbar.update(mini_batch)
        s = sess.run(merged_summary, feed_dict={X:X_train_mini[i], Y:Y_train_mini[i]})
        train_writer.add_summary(s,e)
        s = sess.run(merged_summary_test, feed_dict={X:X_test, Y:Y_test})
        test_writer.add_summary(s,e)
        #s, a, c = sess.run([merged_summary, accuracy, cost], feed_dict={X:X_train, Y:Y_train})
        #train_writer.add_summary(s,e)
        #s = sess.run(merged_summary, feed_dict={X:X_test, Y:Y_test})
        #test_writer.add_summary(s,e)
        #toc=time.time()
        #print("Epoch "  + str(e) +": " + str(c) + "  " + str(round(a*100,1)) + "% " + str(round((toc-tic),1)) + "s")   
        if e % 10 == 0:
            sess.run(assignment, feed_dict={X:X_test, Y:Y_test})
            saver.save(sess,'./test/' + log_string + '/model.ckpt', e)

        #recognition_train, recognition_test = evaluation (Z_Train, Z_Test,batch ,batch_test)
        #print ("Epoch ||  Time  | Cost Traindata |  Recognition Traindata  || Cost Testdata |  Recognition Testndata |")
        #print ("  " + str(e) + "      "+ str(round((toc-tic),1)) + "s          " +str(round(cost_plot_train[(e+1) * num_minibatches-1,0], 2) )   + "                " + str(round(recognition_train,1)) + "%                  "+ str(round(cost_plot_test[(e+1) * num_minibatches-1,0], 2) ) + "                " + str(round(recognition_test,1)) + "%" )
