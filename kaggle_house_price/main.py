# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:05:56 2018

@author: Gabriel
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.contrib.layers as layers


# In[Hypeparameters]: 
epoch = 500
fc_neurons = [80,80] 
weight_decay = 0.1
learning_rate = 0.01


# In[Functions]:
def qualitiesCoding(feature):
    L = feature.shape[0]
    vector = np.zeros((L,1))
    for i in range(L):
        if (feature[i] == "Ex"):
            res = 1
        elif (feature[i] == "Gd"):
            res = 0.8
        elif (feature[i] == "TA"):
            res = 0.6
        elif (feature[i] == "Fa"):
            res = 0.4
        elif (feature[i] == "Po"):
            res = 0.2
        elif (feature[i] == "NA"):
            res = 0
        vector[i,0] = res
    return vector
    

def normalization(feature_train, feature_test):
    mean = feature_train.mean(axis=0)
    std = feature_train.std(axis=0)
    feature_train = (feature_train - mean) / std
    feature_test = (feature_test - mean) / std
    feature_train = feature_train.reshape(feature_train.shape[0],1)
    feature_test = feature_test.reshape(feature_test.shape[0],1)
    return feature_train, feature_test


def one_hot_vector(feature_train, feature_test):
    """
    encoder.fit(feature_train)
    feature_train = encoder.transform(feature_train) 
    #try:
    feature_test =  encoder.transform(feature_test) 
    
    except ValueError:

        print(feature_test)
    
    vectorSize = max(feature_train) + 1
    
    for i in range (feature_train.shape[0]):
        M_tr[i,feature_train[i]] = 1
    for i in range (feature_test.shape[0]):
        M_te[i,feature_test[i]] = 1
    """
    dummy_train = pd.get_dummies(feature_train)
    dummy_new = pd.get_dummies(feature_test)
    dummy_new = dummy_new.reindex(columns = dummy_train.columns, fill_value=0)
    M_tr = dummy_train.values
    M_te = dummy_new.values
    return M_tr, M_te   
    
        
def creatingBatch(csv_train, csv_test, csv_y_test):
    data_train = pd.read_csv(csv_train)
    data_test =  pd.read_csv(csv_test)
    y_train = data_train.loc[:, "SalePrice"].values.reshape(data_train.shape[0], 1)
    data = pd.read_csv(csv_y_test)
    y_test =data.loc[:, "SalePrice"].values.reshape(data_test.shape[0], 1).astype(np.int64)
    features_train= data_train.loc[:,['LotArea', 'Neighborhood', 'OverallQual', 'OverallCond',
                    'ExterQual','ExterCond','BsmtQual', 'KitchenQual','GarageType']]
    features_test= data_test.loc[:,['LotArea', 'Neighborhood', 'OverallQual', 'OverallCond',
                    'ExterQual','ExterCond','BsmtQual', 'KitchenQual','GarageType']]
    Liste_tr = []
    Liste_te = []
    features_train = features_train.replace(np.nan, 'NAN', regex=True) 
    features_test = features_test.replace(np.nan, 'NAN', regex=True) 
    inputNeurons = 0
    for i in range (features_train.shape[1]):
        if (isinstance(features_train.iat[1,i], str)):
            M_tr, M_te = one_hot_vector(features_train.iloc[:,i].values, features_test.iloc[:,i].values)
            Liste_tr.append(M_tr)
            Liste_te.append(M_te)
            inputNeurons += Liste_tr[i].shape[1]
        else:
            f_train, f_test = normalization(features_train.iloc[:,i].values, features_test.iloc[:,i].values)
            Liste_tr.append(f_train)
            Liste_te.append(f_test)
            inputNeurons += 1
    x_train = np.zeros((features_train.shape[0],inputNeurons))  
    x_test = np.zeros((features_test.shape[0],inputNeurons))  
    index = 0
    for i in range (features_train.shape[1]):
        x_train[:, index : index +  Liste_tr[i].shape[1]] = Liste_tr[i]
        x_test[:, index : index +  Liste_tr[i].shape[1]] = Liste_te[i]
        index += Liste_tr[i].shape[1]
    return x_train, y_train, x_test, y_test
    

def create_placeholders(x_data):
    X_p = tf.placeholder(tf.float32, [None, x_data.shape[1]], name = "Inputs")
    Y_p = tf.placeholder(tf.float32, [None, 1], name = "Labels")
    return X_p, Y_p


def Fully_connected(P, fc_neurons):
    L_conec = len(fc_neurons)
    for l in range(0, L_conec):       
        P = tf.contrib.layers.fully_connected(P, 
                                              fc_neurons[l],
                                              scope = 'FC-'+str(l),
                                              weights_regularizer = layers.l2_regularizer(weight_decay),
                                              biases_regularizer=layers.l2_regularizer(weight_decay))
    Z_out = tf.contrib.layers.fully_connected(P, 1, activation_fn=None,
                                              scope = 'FC-'+str(l+1),
                                              weights_regularizer=layers.l2_regularizer(weight_decay),
                                              biases_regularizer=layers.l2_regularizer(weight_decay))
    return Z_out, P


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.square(Y - Z3))
    return cost




def get_var(name,all_vars):
    for i in range(len(all_vars)):
        if all_vars[i].name.startswith(name):
            return all_vars[i]
    return None



# In[Main]:
log_string = 'fc={},lr={},wd={}'.format(fc_neurons, learning_rate, weight_decay)
log_string = log_string.replace(" ", "")
encoder = LabelEncoder()
x_train, y_train, x_test, y_test = creatingBatch("train.csv", "test.csv", "sample_submission.csv")
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(x_train)
    Z_out, embedding_input = Fully_connected(X, fc_neurons)
    with tf.name_scope('cost'):
        cost = compute_cost(Z_out, Y)
        cost_sum = tf.summary.scalar('cost', cost)
    with tf.name_scope('Train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    all_vars = tf.global_variables()
    for l in range(len(fc_neurons) + 1):
        tf.summary.histogram("Biases_FC-"+ str(l), get_var('FC-' + str(l) + '/biases',all_vars))
        tf.summary.histogram("Weight_FC-"+ str(l), get_var('FC-' + str(l) + '/weight',all_vars))
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./tensorboard/train/' + log_string)
    test_writer = tf.summary.FileWriter('./tensorboard/test/' + log_string)
    train_writer.add_graph(sess.graph)
    for e in tqdm(range (epoch)):
        _ , s_tr = sess.run([optimizer, merged_summary], feed_dict={X:x_train, Y:y_train})
        s_te = sess.run(cost_sum, feed_dict={X:x_test, Y:y_test})
        train_writer.add_summary(s_tr,e)
        test_writer.add_summary(s_te,e)
    result = sess.run(Z_out, feed_dict={X:x_train})