#! /bin/env python
#! -*- encoding:utf-8 -*-

"""
# 
# version description: mnist dataset recognition with softmax
# 
# date: 2017.10
# author: michaelzph
#
"""

import time
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# step1: read data
MNIST = input_data.read_data_sets("/mnist-dataset", one_hot=True)


# step2: Definite paramters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25


# step3: Create the placeholder for features and labels
# each image in MNIST data shape is 28*28=784 
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])


# step4: Create the weights and bias
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")


# step5: Predict Y from X, w and b
logits = tf.matmul(X, w) + b


# step6: define the loss
# Use the softmax corss entropy with logits as the loss function
# compute the mean entropy, softmax is applied internally
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy) # compute the mean entropy over examples in the batch


# step7: define training ops
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  n_batches = int(MNIST.train.num_examples / batch_size)
  for i in range(n_epochs): # train the model n_epochs times 
    for _ in range(n_batches):
      X_batch, Y_batch = MNIST.train.next_batch(batch_size)
      _, l_value = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
    if i % 4 == 0:
      w_value, b_value = sess.run([w, b])
      print("the {0}th epoches".format(i))
      #print("weithgs: {}, bias: {}".format(w_value, b_value))
      print("loss: {:.4f}".format(l_value))
  writer = tf.summary.FileWriter("graphs/LR_Mnist", sess.graph)
  
  # test the model
  n_batches = int(MNIST.test.num_examples / batch_size)
  total_correct_pred = 0
  for i in range(n_batches):
    X_batch, Y_batch = MNIST.test.next_batch(batch_size)
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y: Y_batch})
    preds = tf.nn.softmax(logits_batch)
    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    total_correct_pred += sess.run(accuracy)
  print("Accuracy {0}".format(total_correct_pred / MNIST.test.num_examples))






