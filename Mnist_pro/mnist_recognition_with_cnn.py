#! /usr/bin/python
#! -*- encoding:utf-8 -*-


"""
# 
# version description: mnist dataset recognition with cnn
# accurancy 99.2% 
# 
# date: 2017.10
# author: michaelzph
#
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# 1. 定义输入数据并预处理数据
mnist = input_data.read_data_sets("mnist-dataset", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img 

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# 2. 初始化权重与定义网络结构 (3 个conv, 3 个pooling, 1 个fulc, 1 个out)
def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w = init_weight([5, 5, 1, 32])  # patch : 5x5, input : 1, output : 32
w2 = init_weight([5, 5, 32, 64])  # patch : 5x5, input : 32, output : 64
w3 = init_weight([3, 3, 64, 128])  # patch : 3x3, input : 64, output : 128
w4 = init_weight([128 * 4 * 4, 1024])  # fulc 层, input : 128x4x4,

w_o = init_weight([1024, 10]) # output, input : 1024, output : 10


# 定义一个模型函数
# X : 输入数据
# W : 每一层权重
# p_keep_conv, p_keep_hidden:dropout 要保留的神经元比例

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # 第一组卷积层和池化层,最后dropout一些神经元
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    # l1a shape=(?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l1  shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)
    
    
    # 第二组卷积层和池化层,最后dropout一些神经元
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    # l2a shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l2 shape=(?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    
    # 第三组卷积层和池化层,最后dropout一些神经元
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    # l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
    l3 =  tf.nn.dropout(l3, p_keep_conv)
    
    
    # 全连接层,最后dropout一些神经元
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    
    
    # 输出层
    pyx = tf.matmul(l4, w_o)
    return pyx  # 返回预测值



p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)  # 得到预测值

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)
predict_op = tf.argmax(py_x, 1)

correct_prediction = tf.equal(tf.arg_max(py_x, 1), tf.arg_max(Y, 1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 128
test_size = 256

st = time.clock()
# 开始训练和评估
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("graphs/", sess.graph)
    
    for i in range(100):
        st_p_batch = time.clock()
        training_batch = zip(range(0, len(trX), batch_size), 
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
        train_acc = accurancy.eval(feed_dict={X: trX[start:end], Y:trY[start:end], 
                                          p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print(i, "train accurancy: {:.6f}".format(train_acc))
            
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        test_acc = accurancy.eval(feed_dict={X: teX[test_indices], Y: teY[test_indices], 
                                          p_keep_conv: 1.0, p_keep_hidden: 1.0})
        print(i, "test accurancy: {:.6f}".format(test_acc))


        et_p_batch = time.clock()
        print("{} batch cost {:.6f} s".format(i, (et_p_batch-st_p_batch)))
    
et = time.clock()

print("training cost {:.6f} s.".format(et-st))










