# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:00:28 2017

@author: HOSSAM ABDELHAMID
"""


# coding: utf-8

# # DATA READER

# JPEG Compression: Quality factor estimation
# ---------------------
# Quality factor estimation given known candidate set. In thisexperiment,weassumethattheinvestigatorknowsthatthe forger used one of quality factor values in a fixed set. Here, this set is Θ = {70,90,U}. Our estimate θ is the quality factor denoted by QF,   
# and U is no compression case.                                                                                                    
# QF=70-----> Class:0                                                                                                              
# QF=90-----> Class:1                                                                                                                      
# U------------> Class:2

# Included Libraries
# --------

# In[1]:


import scipy.misc
import random
import csv
DATA_DIR = '../Database/'
import tensorflow as tf
import numpy as np
Classes=5
# One Hot Matrix
# --

# In[2]:


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C,axis=1)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot


# Data Handling, shuffling and loading
# --

# In[54]:


class DataReader(object):
    def __init__(self, data_dir=DATA_DIR,sequential=False):
        self.load()

    def load(self):
        xs = []     #input data
        ys = []     #output Label 
        yss=[]
        self.train_batch_pointer = 0 #pointer for taking mini batch one after another
        self.test_batch_pointer = 0
        self.total = 0  # Number of training samples
        # CVS file that has all images names, Class and quality factor
        with open('scale05.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching scale05 ...")
            for row in reader:
                xs.append(DATA_DIR+row['factor']+'/'+row['image'])
                yss.append((row['classes']))
                self.total += 1
        with open('scale08.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching scale08 ...")
            for row in reader:
                xs.append(DATA_DIR+row['factor']+'/'+row['image'])
                yss.append((row['classes']))
                self.total += 1
        with open('scale1.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching scale1 ...")
            for row in reader:
                xs.append(DATA_DIR+row['factor']+'/'+row['image'])
                yss.append((row['classes']))
                self.total += 1
        with open('scale15.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching scale15 ...")
            for row in reader:
                xs.append(DATA_DIR+row['factor']+'/'+row['image'])
                yss.append((row['classes']))
                self.total += 1
        with open('scale2.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching scale2 ...")
            for row in reader:
                xs.append(DATA_DIR+row['factor']+'/'+row['image'])
                yss.append((row['classes']))
                self.total += 1
                
        print('Total training data: ' + str(self.total))
        ys=one_hot_matrix(np.float32(yss),Classes)
        self.num_images = len(xs)
        c = list(zip(xs, ys))
        random.shuffle(c)
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        self.train_xs, self.train_ys = zip(*c)
        self.train_ys=np.float32(self.train_ys)
        print(self.train_ys.shape)
        xtest = []     #input data
        ytest = []     #output Label 
        ysstest=[]
        self.total_test = 0
        # CVS file that has all images names, Class and quality factor
        with open('TEST.csv') as f:
            reader = csv.DictReader(f)
            print("Fetching Testing Data ...")
            for row in reader:
                xtest.append(DATA_DIR +row['factor']+'/'+row['image'])
                ysstest.append((row['classes']))
                self.total_test += 1
        print('Total test data: ' + str(self.total_test))
        ytest=one_hot_matrix(np.float32(ysstest),Classes)
        c = list(zip(xtest, ytest))
        # Random Data xs->images , ys->ouptut labels one hot encoded matrix 
        random.shuffle(c)
        self.test_xs, self.test_ys = zip(*c)
        self.test_ys=np.float32(self.test_ys)
        print(self.test_ys.shape)
        # Get Random mini batch of size batch_size
    def load_train_batch(self, batch_size):
        x_out = np.zeros((batch_size,256,256,1))
        y_out = np.zeros((batch_size,Classes))
        image = np.zeros((256,256,1))
        for i in range(0, batch_size):
            image = scipy.misc.imread(self.train_xs[(self.train_batch_pointer + i) % self.num_images])
            image=image.reshape(256,256,1)
            x_out[i,:,:,:]=image / 255.0
            y_out[i,:]=self.train_ys[(self.train_batch_pointer + i) % self.num_images]
        self.train_batch_pointer += batch_size
        return x_out, y_out
    
    def load_test_data(self, test_size):
        x_out = np.zeros((test_size,256,256,1))
        y_out = np.zeros((test_size,Classes))
        image = np.zeros((256,256,1))
        for i in range(0, test_size):
            image = scipy.misc.imread(self.test_xs[(self.test_batch_pointer + i) % self.total_test])
            image=image.reshape(256,256,1)
            x_out[i,:,:,:]=image / 255.0
            y_out[i,:]=self.test_ys[(self.test_batch_pointer + i) % self.total_test]
        self.test_batch_pointer += test_size
        return x_out, y_out


