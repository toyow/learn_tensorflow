from model import *
from matplotlib import pyplot as plt
import numpy as np
file_dir='C:\\toyow\\software\\Anaconda3\\envs\\cat vs dog\\data\\test1\\test1\\000.jpg'
image_size=208

#读取图片，解码，归一化，增加一个维度
image=tf.read_file(file_dir)
image_train = tf.image.decode_jpeg(image, channels=3)
image_train = tf.image.resize_images(image_train, [image_size, image_size])
image_train = tf.cast(image_train, tf.float32) / 255.
image_train=tf.expand_dims(image_train,0)
#喂进网络，得到预测结果的tensor
prediction=inference(image_train,2)
prediction=tf.nn.softmax(prediction)
saver=tf.train.Saver()

with tf.Session() as sess:
    #先初始化再载入权重
    sess.run(tf.global_variables_initializer())

    #恢复权重，run出结果
    saver.restore(sess,tf.train.latest_checkpoint('./logs/'))
    image,logit=sess.run([image_train,prediction])
    max_index=np.argmax(logit)
    if max_index == 0:
        label = '%.2f%% is a cat.' % (logit[0][0] * 100)
    else:
        label = '%.2f%% is a dog.' % (logit[0][1] * 100)

    plt.imshow(image[0])
    plt.title(label)
    plt.show()
