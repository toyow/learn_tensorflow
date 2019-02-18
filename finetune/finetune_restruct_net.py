from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf
import tensorflow.contrib.slim as slim

batch_size=100
batch_num=mnist.train.num_examples//batch_size

def bias_variabel(name,shape,initializer):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=initializer)
def weights_variabel(name,shape,std):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=std,dtype=tf.float32))


def inference(x):
    with tf.variable_scope('layer1') as scope:
        weights=weights_variabel('weights',[784,256],0.04)
        bias=bias_variabel('bias',[256],tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.add(tf.matmul(x,weights),bias),name=scope.name)
    with tf.variable_scope('layer2') as scope:
        weights=weights_variabel('weights',[256,128],0.02)
        bias=bias_variabel('bias',[128],tf.constant_initializer(0.0))
        layer2=tf.nn.relu(tf.add(tf.matmul(layer1,weights),bias),name=scope.name)
        # layer2=tf.stop_gradient(layer2,name='layer2_stop')#layer2及其以前的op均不进行反向传播
    with tf.variable_scope('layer3') as scope:
        weights=weights_variabel('weights',[128,64],0.001)
        bias=bias_variabel('bias',[64],tf.constant_initializer(0.0))
        layer3=tf.nn.relu(tf.add(tf.matmul(layer2,weights),bias),name=scope.name)
    with tf.variable_scope('softmax_linear_1') as scope:
        weights = weights_variabel('weights', [64, 10], 0.0001)
        bias = bias_variabel('bias', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(layer3, weights), bias,name=scope.name)
    return softmax_linear


def loss(labels,logits):
    print(labels.get_shape().as_list())#免得形状对不上
    print(logits.get_shape().as_list())
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',cross_entropy_mean)
    return cross_entropy_mean

def train():
#恢复原网络的op tensor
    with tf.Graph().as_default() as g:

        with tf.name_scope('input'):
            x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
            y_input = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

        softmax_linear=inference(x_input)#继续前向传播
        cost=loss(y_input,softmax_linear)
        train_op=tf.train.AdamOptimizer()
        grads=train_op.compute_gradients(cost)#返回的是(gradent,varibel)元组对的列表

        variables_low_LR = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:4]#获取低学习率的变量列表

        low_rate=0.0001
        high_rate=0.001

        new_grads_varible=[]#新的列表

        for grad in grads:#对属于低学习率的变量的梯度，乘以一个低学习率
            if grad[1] in variables_low_LR:
                new_grads_varible.append((low_rate*grad[0],grad[1]))
            else:
                new_grads_varible.append((high_rate * grad[0], grad[1]))

        apply_gradient_op = train_op.apply_gradients(new_grads_varible)

        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_linear,1),tf.argmax(y_input,1)),tf.float32))

        variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[:4]
        saver = tf.train.Saver(variables_to_restore)

    with tf.Session(graph=g) as sess:
        #首先恢复权重
        saver.restore(sess,save_path=tf.train.latest_checkpoint('./my_ckpt_save_dir/'))
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        for epoch in range (50):
            for _ in range(batch_num):
                x_train_batch,y_train_batch=mnist.train.next_batch(batch_size)
                sess.run(apply_gradient_op,feed_dict={x_input:x_train_batch,y_input:y_train_batch})
                cost_value= sess.run(cost,feed_dict={x_input:x_train_batch,y_input:y_train_batch})
            accuracy_value=sess.run(accuracy,feed_dict={x_input:mnist.test.images,y_input:mnist.test.labels})
            print(("%s epoch: %d,loss:%.6f accuracy:%.6f")%(datetime.now(),epoch+1,cost_value,accuracy_value))
            if (epoch+1)%5==0:
                check_point_dir=os.path.join('my_ckpt_restore_dir','wdy_finetune_model')
                saver.save(sess,check_point_dir,global_step=epoch+1)

train()



