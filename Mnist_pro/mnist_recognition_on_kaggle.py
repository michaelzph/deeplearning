#! /bin/env python
#! -*- encoding:utf-8 -*-


import numpy as np
import pandas as pd
import tensorflow as tf
import time


# image data file
trainfile = "kaggle-data/train.csv"
testfile = "kaggle-data/test.csv"

# read data
images = pd.read_csv(trainfile)


# data process
img_data = images.iloc[:, 1:].values
img_data = img_data.astype(np.float32)

# image size, width and height of each image 
img_size = img_data.shape[1]
img_width = img_height = np.ceil(np.sqrt(img_data.shape[1])).astype(np.uint8)


# label data
label_data = images.label.values.ravel()

# label counts 
label_counts = np.unique(label_data).shape[0]

# define input and output
X_input = tf.placeholder(tf.float32, shape=[None, img_size])
y_output = tf.placeholder(tf.float32, shape=[None, label_counts])

# define dropout in conv and fuc
keep_conv = tf.placeholder(tf.float32)
keep_fuc = tf.placeholder(tf.float32)

# dev set counts
DEV_SET_NUM = 2000


# convert class labels from scales to one-hot vector
# 0: [1 0 0 0 0 0 0 0 0 0]
# 1: [0 1 0 0 0 0 0 0 0 0]
# input: 
#   label_data--label of each line (42000)
#   classes_counts--num of classes (10)
# return:
#   label_vector: (42000,10)
# 
def dense_to_one_hot_vector(label_data, class_counts):
    label_num = label_data.shape[0]
    index_offset = np.arange(label_num) * class_counts 
    label_one_hot = np.zeros((label_num, class_counts))
    label_one_hot.flat[index_offset + label_data.ravel()] = 1
    return label_one_hot


labels = dense_to_one_hot_vector(label_data, label_counts)


# split train data into train-set and dev-set
def train_and_dev_dataset(data, lables, dev_counts):
    dev_data = data[:DEV_SET_NUM]
    train_data = data[DEV_SET_NUM:]
    dev_label = labels[:DEV_SET_NUM]
    train_label = labels[DEV_SET_NUM:]
    return (train_data, train_label, dev_data, dev_label)


# get train data, train label, dev data and dev label
train_data, train_label, dev_data, dev_label = train_and_dev_dataset(img_data, labels, DEV_SET_NUM)


# initialize the weight and bias
#def init_weight(input_size, output_size):
#    return tf.Variable(tf.truncated_normal(input_size, stddev=0.01) / tf.sqrt(input_size / 2), name='weights')
def init_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32), name='weights')


#def init_bias(output_size):
#    return tr.Variable(tf.constant(output, 0.1), name='biases')
def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, dtype=tf.float32), name='biases')


# convolution layer
def conv2d(X, w):
    return tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME')

# pooling layer
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# batch_normalization 


# dropout 
def do_dropout(last_out, keep_prob):
    return tf.nn.dropout(last_out, keep_prob=keep_prob)




epochs_completed = 0
index_in_epoch = 0
num_examples = img_data.shape[0]

# create next batch for training
def next_batch(batch_size):
    global epochs_completed
    global index_in_epoch 
    global num_examples
    
    start = index_in_epoch 
    if start + batch_size > num_examples:
        epochs_completed += 1
        # get the rest examples in this epoch
        rest_num_examples = num_examples - batch_size
        rest_image_part_1 = train_data[start: rest_num_examples]
        rest_label_part_1 = train_label[start: rest_num_examples]
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        index_in_epoch = batch_size - rest_num_examples
        rest_image_part_2 = train_data[0: index_in_epoch]
        rest_label_part_2 = train_label[0: index_in_epoch]
        image_new = np.concatenate((rest_image_part_1, rest_image_part_2), axis=0)
        label_new = np.concatenate((rest_label_part_1, rest_label_part_2), axis=0)
        return image_new, label_new 
    else:
        start = index_in_epoch
        end = index_in_epoch + batch_size
        image = train_data[start:end]
        label = train_label[start:end]
        return image, label

    
# build cnn for mnist
def build_cnn(X, y, img_height, img_width, keep_conv, keep_fuc):
    with tf.name_scope("reshape"):
        image_input = tf.reshape(X, [-1, img_height, img_width, 1])
    # layer_1
    with tf.name_scope("conv1"):
        w_conv1 = init_weight([5, 5, 1, 32]) 
        b_conv1 = init_bias([32])
        h_conv1 = tf.nn.relu(conv2d(image_input, w_conv1) + b_conv1)
    with tf.name_scope("pooling1"):
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.name_scope("dropout1"):
        h_dropout1 = do_dropout(h_pool1, keep_prob=keep_conv)
    
    # layer_2 
    with tf.name_scope("conv2"):
        w_conv2 = init_weight([3, 3, 32, 64])
        b_conv2 = init_bias([64])
        h_conv2 = tf.nn.relu(conv2d(h_dropout1, w_conv2) + b_conv2)
    
    with tf.name_scope("pooling2"):
        h_pool2 = max_pool_2x2(h_conv2)
    
    with tf.name_scope("dropout2"):
        h_dropout2 = do_dropout(h_pool2, keep_prob=keep_conv)
    
    # layer_3 
    with tf.name_scope("conv3"):
        w_conv3 = init_weight([3, 3, 64, 128])
        b_conv3 = init_bias([128])
        h_conv3 = tf.nn.relu(conv2d(h_dropout2, w_conv3) + b_conv3)
        
    with tf.name_scope("pooling3"):
        h_pool3 = max_pool_2x2(h_conv3)
    
    with tf.name_scope("dropout3"):
        h_dropout3 = do_dropout(h_pool3, keep_prob=keep_conv)
    
    # fuc1 
    with tf.name_scope("fuc1"):
        w_fuc1 = init_weight([128*4*4, 1024])
        b_fuc1 = init_bias([1024])
        h_fuc1_flat = tf.reshape(h_dropout3, [-1, 128*4*4])
        h_fuc1 = tf.nn.relu(tf.matmul(h_fuc1_flat, w_fuc1) + b_fuc1)
        h_fuc1_drop = do_dropout(h_fuc1, keep_prob=keep_fuc)
        
    # fuc2
    with tf.name_scope("fuc2"):
        w_fuc2 = init_weight([1024, 10])
        b_fuc2 = init_bias([10])
        h_fuc2 = tf.nn.relu(tf.matmul(h_fuc1_drop, w_fuc2) + b_fuc2)
        out = do_dropout(h_fuc2, keep_prob=keep_fuc)
        
    return out



# train the model
def training():
    
    y_pred = build_cnn(X_input, y_output, img_height, img_width, keep_conv, keep_fuc)
    print("type of y_pred --> {}".format(y_pred))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_output))
    optimization = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)
    correct_pred = tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    batch_size = 128
    dev_size = 256
    iteration = 100

    epochs = int(train_data.shape[0] / batch_size)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(iteration):
            s_time = time.clock()

            for epo in range(epochs):
	                
                train_x, train_y = next_batch(batch_size)
                sess.run(optimization, feed_dict={X_input:train_x, y_output:train_y, keep_conv:0.5, keep_fuc:1.0})
                train_accuracy = accuracy.eval(feed_dict={X_input:train_x, y_output:train_y, keep_conv: 0.5, keep_fuc: 1.0})
                tf.summary.scalar("train accuracy", train_accuracy)
            
                dev_x, dev_y = next_batch(dev_size)
                dev_accuracy = accuracy.eval(feed_dict={X_input:dev_x, y_output:dev_y, keep_conv: 0.5, keep_fuc: 1.0})
                tf.summary.scalar("dev accuracy", dev_accuracy)

                print("epoch {0}: train accuracy / dev accuracy ----> {1:.6f} / {2:.6f}".format(epo, train_accuracy, dev_accuracy))

                tf.summary.FileWriter("log/train", sess.graph)
                e_time = time.clock()
            print("iter {0}: train accuracy / dev accuracy ----> {1:.6f} / {2:.6f}".format(i, train_accuracy, dev_accuracy))
            print("each iter cost: {:.6f}".format(e_time - s_time))
            


if __name__=="__main__":

    training()


