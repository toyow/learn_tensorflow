from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import os
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf

batch_size=100
batch_num=mnist.train.num_examples//batch_size

def bias_variabel(name,shape,initializer):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=initializer)
def weights_variabel(name,shape,std):
    return tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=std,dtype=tf.float32))


def inference(input_tensor):
    with tf.variable_scope('layer3') as scope:
        weights=weights_variabel('weights',[128,64],0.001)
        bias=bias_variabel('bias',[64],tf.constant_initializer(0.0))
        layer3=tf.nn.relu(tf.add(tf.matmul(input_tensor,weights),bias),name=scope.name)
    with tf.variable_scope('softmax_linear_2') as scope:
        weights = weights_variabel('weights', [64, 10], 0.0001)
        bias = bias_variabel('bias', [10], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(layer3, weights), bias,name=scope.name)
    return softmax_linear


def loss(labels,logits):
    print(labels.get_shape().as_list())#免得形状对不上
    print(logits.get_shape().as_list())
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_2')
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss',cross_entropy_mean)
    return cross_entropy_mean

def train():
#恢复原网络的op tensor
    with tf.Graph().as_default() as g:
        saver=tf.train.import_meta_graph('./my_ckpt_save_dir/wdy_model-15.meta')#把原网络载入到图中

        x_input=g.get_tensor_by_name('input/x:0')#恢复原op的tensor
        y_input = g.get_tensor_by_name('input/y:0')
        layer2=g.get_tensor_by_name('layer2/layer2:0')
        #layer2=tf.stop_gradient(layer2,name='layer2_stop')#layer2及其以前的op均不进行反向传播

        softmax_linear=inference(layer2)#继续前向传播
        cost=loss(y_input,softmax_linear)



        train_op=tf.train.AdamOptimizer(0.001,name='Adma2').minimize(cost)

        accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax_linear,1),tf.argmax(y_input,1)),tf.float32))
    with tf.Session(graph=g) as sess:
        #首先恢复权重
        saver.restore(sess,save_path=tf.train.latest_checkpoint('./my_ckpt_save_dir/'))
        saver2=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epoch in range (10):
            for _ in range(batch_num):
                x_train_batch,y_train_batch=mnist.train.next_batch(batch_size)
                sess.run(train_op,feed_dict={x_input:x_train_batch,y_input:y_train_batch})
                cost_value= sess.run(cost,feed_dict={x_input:x_train_batch,y_input:y_train_batch})
            accuracy_value=sess.run(accuracy,feed_dict={x_input:mnist.test.images,y_input:mnist.test.labels})
            print(("%s epoch: %d,loss:%.6f accuracy:%.6f")%(datetime.now(),epoch+1,cost_value,accuracy_value))
            if (epoch+1)%5==0:
                check_point_dir=os.path.join('111','wdy_finetune_model')
                saver2.save(sess,check_point_dir,global_step=epoch+1)

train()



