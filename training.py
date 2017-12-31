# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:45:41 2017

@author: HOSSAM ABDELHAMID
"""


# coding: utf-8

# # A Generic Approach Towards Image Manipulation Parameter Estimation Using Convolutional Neural Networks
# 
# 

# Reference:Belhassen Bayar , Matthew C. Stamm , A Generic Approach Towards Image Manipulation Parameter
# Estimation Using Convolutional Neural Networks.Proceedings of the 5th ACM Workshop [DOI: 10.1145/3082031.3083249].

# Included Libraries
# ---

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import time
import tempfile
from cnn_model import ConvModel ,pred_error_filter
from data_read import DataReader  
import numpy as np
import matplotlib.pyplot as plt

# CNN Model Configuration
# -------------

# When training each CNN, we set the batch size equal to 64 and the parameters of the stochastic gradient descent as follows: momentum = 0.95,decay = 0.0005, and a learning rate ϵ = 10−3 that decreases every 3 epochs,which is the number of times that every sample in the training data was trained,by a factorγ = 0.5. We trained the CNN in each experiment for 36 epochs. Note that training and testing are disjoint and CNNs were tested on separate testing datasets. Additionally,while training CNNs,their testing accuracies on a separate testing data base were recorded every 1,000 iterations to produce tables in this section

# In[2]:


'''Size of Batch'''
BATCH_SIZE = 64
TEST_SIZE= 64
'''Steps NUM = epoch_num*batch_num'''
EPOCH_NUM=30
#start iteration
START_STEP = 0
'''Momentum'''
MOMENTUM = 0.95
'''Learning Rate Every 3 epoches by factor of 0.5''' 
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_FACTOR=0.5
NUM_EPOCHS_PER_DECAY=6
'''Saving Model'''
LOGDIR = 'vol'
CHECKPOINT_EVERY = 100
RESTORE_FROM=None
print_cost = True
# In[3]:
# used Functions
def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)
    ### END CODE HERE ###
    return cost

# Training Environment
# -------------------

# In[4]:


def main():
    print("Start training Model...")
    #CNN_Model
    '''initialization'''
    tf.reset_default_graph()
    sess = tf.InteractiveSession()  
    # Forward propagation      
    model = ConvModel()
    costs = []                                        # To keep track of the cost
    learning_rate=[]
    # obtain many different residual features so we should force the first
    #layer parameters after each iteration
    prederr=pred_error_filter([5,5,1, 5])
    #We define global_step as a variable initialized at 0
    global_step=tf.Variable(0,trainable=False)
   # train_vars = tf.trainable_variables()
    # Cost function
    with tf.name_scope('loss'):
        loss = compute_cost(model.y,model.y_)
    loss = tf.reduce_mean(loss)# tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * l2_reg
    #data reading
    data_reader = DataReader()
    #calculate number of iterations per epoch
    NUM_BATCHES_PER_EPOCH=int(data_reader.num_images/BATCH_SIZE)
    print('Num of batches per epoch :',NUM_BATCHES_PER_EPOCH)
    NUM_TEST_DATA=int(data_reader.total_test/TEST_SIZE)
    NUM_STEPS=NUM_BATCHES_PER_EPOCH*EPOCH_NUM
    print("Total No. of iterations :",NUM_STEPS)
    #decay of learning rate
    decay_steps=int(NUM_BATCHES_PER_EPOCH*NUM_EPOCHS_PER_DECAY)
    decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
    with tf.name_scope('momentum_optimizer'):
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,global_step=global_step)
    #prediction
    correct_prediction = tf.equal(tf.argmax(model.y,1), tf.argmax(model.y_,1))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
           
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    #intialize saving
    saver = tf.train.Saver()
    #momentum optimizer

    min_loss = 1.0
    start=0
    init = tf.global_variables_initializer()
    sess.run(init)
    steps=0
  #restoring the model
    if RESTORE_FROM is not None:
        saver.restore(sess, os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
        print('Model restored from ' + os.getcwd()+'\\'+LOGDIR+'\\'+RESTORE_FROM)
       
    #tf.summary.scalar("loss", loss)
   # merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(LOGDIR,sess.graph)

    test_error=0
    for epoch in range(EPOCH_NUM):
        minibatch_cost = 0.0
        for i in range(START_STEP, NUM_BATCHES_PER_EPOCH):
            start = time.time()
            steps+=1
            #get minibatch
            xs, ys = data_reader.load_train_batch(BATCH_SIZE)
            # run optimizer and loss
            _ , temp_cost = sess.run([train_step,loss], feed_dict={model.x: xs, model.y_: ys})
            # force fiirst filter to be prediction error filter
            assign=model.W_conv1.assign(prederr)
            sess.run(assign)
            #evauate train error
            train_error = loss.eval(feed_dict={model.x: xs, model.y_: ys})
            #evaluate average error per epoch
            minibatch_cost += temp_cost / NUM_BATCHES_PER_EPOCH
            #evaluate train accuracy
            train_accuracy = accuracy.eval({model.x: xs, model.y_: ys})
            end = time.time()
            elapsed = end - start
            
            print("Step%d [Learning Rate= %f ,Train Loss= %g ,Accuracy= %g ,Elapsed Time= %g Sec/minibatch ,Estimated Training Time= %g min]"  % (steps, decayed_learning_rate.eval(),train_error,train_accuracy*100,elapsed,elapsed*(NUM_STEPS-steps)/60))
            
        #summary, _ = sess.run([merged_summary_op, train_step], feed_dict={model.x: xs, model.y_: ys})
        #summary_writer.add_summary(summary, i)
        #test every 300 iteration
            if steps% 1000 == 0 or steps==NUM_BATCHES_PER_EPOCH*EPOCH_NUM-1:
                test_cost = 0.0
                test_acc=0.0
                for j in range(NUM_TEST_DATA):
                    xtest, ytest = data_reader.load_test_data(TEST_SIZE)
                    test_error = loss.eval(feed_dict={model.x: xtest, model.y_: ytest})
                    test_cost +=  test_error/ NUM_TEST_DATA
                    test_accuracy = accuracy.eval({model.x: xtest, model.y_: ytest})
                    test_acc +=  test_accuracy/ NUM_TEST_DATA
                print("Testing... Test Loss= %g  Accuracy:= %g" % (test_cost,test_acc*100))
            #saving every 100 iteration
            if steps > 0 and steps % CHECKPOINT_EVERY == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, test_error))
                filename = saver.save(sess, checkpoint_path)
                print("Model saved in file: %s" % filename)
                if test_error < min_loss:
                    min_loss = test_error
                    if not os.path.exists(LOGDIR):
                        os.makedirs(LOGDIR)
                    checkpoint_path = os.path.join(LOGDIR, "model-step-%d-val-%g.ckpt" % (i, test_error))
                    filename = saver.save(sess, checkpoint_path)
                    print("Model saved in file: %s" % filename)
    #save_path = saver.save(sess, model_path)
    #print("Model saved in file: %s" % save_path)
                    # Print the cost every epoch
        
        learning_rate.append(decayed_learning_rate.eval())
        if print_cost == True and epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if print_cost == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

        
    checkpoint_path = os.path.join(LOGDIR, "model-step-final.ckpt")
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.show()  
    
    plt.plot(np.squeeze(learning_rate))
    plt.ylabel('Learning rate')
    plt.xlabel('iterations (per tens)')
    plt.show() 
if __name__ == '__main__':
    main()

