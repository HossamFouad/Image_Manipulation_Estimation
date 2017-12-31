# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:39:17 2017

@author: HOSSAM ABDELHAMID
"""


# coding: utf-8

# # Deep Learning Model
# 
# 
# ___________________________________________________________________________

# This file includes The "`Convolution Neural Network Architecture`" for Image Manipulation Parameter
# Estimation.
# _______________
# 
# Implementation the ConvNet model from the paper :
# Belhassen Bayar , Matthew C. Stamm , A Generic Approach Towards Image Manipulation Parameter
# Estimation Using Convolutional Neural Networks.Proceedings of the 5th ACM Workshop [DOI: 10.1145/3082031.3083249].

# 
# ![CNN_MODEL.png](attachment:CNN_MODEL.png)

# In[3]:


import tensorflow as tf
import numpy as np
CLASS_NUM=5


# In[5]:


#Weight initialization Shape is list of [Length Width Depth Num_filters]
def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    initial = initializer(shape=shape)
    return tf.Variable(initial)


#2D CONV that takes input , weights and stride and return it is output
def conv2d(x, W, stride,pad):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=pad)
# inialize prediction error filter with Shape which is list of [Length Width Depth Num_filters]
def pred_error_filter(shape):
    middle_l=round(shape[0]/2)
    middle_w=round(shape[1]/2)
    filter_err=np.ones(shape)
    filter_err[middle_l,middle_w,:,:]=-1.0
    initial = tf.Variable(filter_err, dtype=tf.float32)
    return initial

class ConvModel(object):
    
    def __init__(self,batch_norm=True, whitening=False, is_training=True):
        '''Input:x is grayscale images with size=256*256'''
        '''Output:y_ has CLASS_NUM dimensions which are equal the number of subsets available for estimation'''
        self.x = tf.placeholder(tf.float32, shape=[None, 256, 256,1], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])
        
        x_image = self.x
        '''Prediction Error Feature Extraction'''
        '''Constrained Conv'''
        '''5@5*5*1 prediction error filters: CONV(stride=1)''' 
        with tf.name_scope('conv1'):
            self.W_conv1 = weight_variable([5,5,1, 5])
            self.h_conv1 = conv2d(x_image, self.W_conv1, 1,pad='VALID') 
        
##########################################################################################################
##########################################################################################################
        '''Hierarchical Feature Extraction'''
        '''Conv2'''
        '''96@7*7*5 filters: CONV (stride=2)-> Batch Norm -> TanH -> max pooling (3*3,stride=2)'''
        #Initialize
        with tf.name_scope('conv2'):
            self.W_conv2 = weight_variable([7, 7, 5, 96])
        #CONV (stride=2)
            self.h_conv2 = conv2d(self.h_conv1, self.W_conv2, 2,pad='SAME')
        #Batch Norm
            self.h_conv2 = tf.contrib.layers.batch_norm(self.h_conv2, is_training=is_training, trainable=True)
        #TanH
            self.h_conv2 = tf.nn.tanh(self.h_conv2)
        
        #max pooling (3*3,stride=2)
        with tf.name_scope('pool1'):
            self.pool1 =tf.nn.max_pool(self.h_conv2, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
        
        ##########################################################################
        '''Conv3'''
        '''64@5*5*96 filters: CONV (stride=1)-> Batch Norm -> TanH -> max pooling (3*3,stride=2)'''
        #Initialize
        with tf.name_scope('conv3'):
            self.W_conv3 = weight_variable([5, 5, 96, 64])
        #CONV (stride=1)
            self.h_conv3 = conv2d(self.pool1, self.W_conv3, 1,pad='SAME') 
        #Batch Norm
            self.h_conv3 = tf.contrib.layers.batch_norm(self.h_conv3, is_training=is_training, trainable=True)
        #TanH
            self.h_conv3 = tf.nn.tanh(self.h_conv3)
        
        #max pooling (3*3,stride=2)
        with tf.name_scope('pool2'):
            self.pool2 =tf.nn.max_pool(self.h_conv3, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME')
        
        ##########################################################################
        '''Conv4'''
        '''64@5*5*64 filters: CONV (stride=1)-> Batch Norm -> TanH -> max pooling (3*3,stride=2)'''
        #Initialize
        with tf.name_scope('conv4'):
            self.W_conv4 = weight_variable([5, 5, 64, 64])
        #CONV (stride=1)
            self.h_conv4 = conv2d(self.pool2, self.W_conv4, 1,pad='SAME') 
        #Batch Norm
            self.h_conv4 = tf.contrib.layers.batch_norm(self.h_conv4, is_training=is_training, trainable=True)
        #TanH
            self.h_conv4 = tf.nn.tanh(self.h_conv4)
       
        #max pooling (3*3,stride=2)
        with tf.name_scope('pool3'):
            self.pool3 =tf.nn.max_pool(self.h_conv4, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'VALID')
      
##########################################################################################################
##########################################################################################################
        '''Cross Feature Maps Learning'''
        '''Conv5'''
        '''128@1*1*64 filters: CONV (stride=1)-> Batch Norm -> TanH -> AVG pooling(3*3,stride=2)'''
        #Initialize
        with tf.name_scope('conv5'):
            self.W_conv5 = weight_variable([1, 1, 64, 128])
        #CONV (stride=1)
            self.h_conv5 = conv2d(self.pool3, self.W_conv5, 1,pad='SAME')
        #Batch Norm
            self.h_conv5 = tf.contrib.layers.batch_norm(self.h_conv5, is_training=is_training, trainable=True)
        #TanH
            self.h_conv5 = tf.nn.tanh(self.h_conv5)
        
        #AVG pooling (3*3,stride=2)
        with tf.name_scope('pool4'):
            self.pool4 =tf.nn.avg_pool(self.h_conv5, ksize = [1,3,3,1], strides = [1,2,2,1], padding = 'SAME') 
        
##########################################################################################################
##########################################################################################################
        '''Classification'''
        '''FC1:200'''
        with tf.name_scope('fc1'):
            self.W_fc1 = weight_variable([8192, 200])
            self.pool4_flat =tf.contrib.layers.flatten(self.pool4)
            self.h_fc1 = tf.nn.tanh(tf.matmul(self.pool4_flat, self.W_fc1),name='fc1')
       
        ##########################################################################
        '''FC2:200'''
        with tf.name_scope('fc2'):
            self.W_fc2 = weight_variable([200, 200])
            self.h_fc2 = tf.nn.tanh(tf.matmul(self.h_fc1, self.W_fc2) , name='fc2')
        
        ##########################################################################
        '''FC3:CLASS_NUM''' 
        with tf.name_scope('fc3'):
            self.W_fc3 = weight_variable([200, CLASS_NUM])
            y = tf.matmul(self.h_fc2, self.W_fc3,name='y' ) 
        self.y = y 
        
        

    

