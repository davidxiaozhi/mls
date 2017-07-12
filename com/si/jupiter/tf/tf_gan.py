# -*- coding:UTF-8 -*-
"""
    官方文档上的例子(https://www.tensorflow.org/get_started/get_started),介绍一下tensorflow 的一些基本概念
    tensorflow　当中的计算图是将一系列的tensorflow操作以图节点的方式进行分类存储从而构成图
    A computational graph is a series of TensorFlow operations arranged into a graph of nodes. 
    Let's build a simple computational graph. Each node takes zero or more tensors as inputs
     and produces a tensor as an output. 
     One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores
      internally. We can create two floating point Tensors node1 and node2 as follows:
"""

"""
    GAN 对抗神经网络的核心 是一个 伪造者generator 和一个鉴定器 discriminator 彼此相互训练
    matplotlib (2.0.2)
    numpy (1.13.1)
    Pillow (4.2.1)
    tensorflow (1.2.1)

"""
import tensorflow as tf
import os
import sys
import numpy as np
from PIL import ImageOps, Image  # python 图片处理包

# Set LOAD to True to load a trained model or set it False to train a new one.

LOAD = False

# Dataset directories

DATASET_PATH = './Dataset/Roses/'

DATASET_CHOSEN = 'roses'  # required by utils.py -> ['birds', 'flowers', 'black_birds']

# Model hyperparameters

Z_DIM = 100  # The input noise vector dimension

BATCH_SIZE = 12

N_ITERATIONS = 30000

LEARNING_RATE = 0.0002

BETA_1 = 0.5

IMAGE_SIZE = 64  # Change the Generator model if the IMAGE_SIZE needs to be changed to a different value



def load_dataset(path, data_set='birds', image_size=64):
    """

    加载图片从指定目录,将它们重构为64 * 64，并将其缩放值设置为-1和1之间，以预处理这些图像

    :param path:  数据集所在目录

    :param data_set: 'birds' -> loads data from birds directory, 'flowers' -> loads data from the flowers directory.

    :param image_size: 返回数组的尺寸大小

    :return: numpy array, shape : [图片数量,图片尺寸,图片尺寸,3]


    """
    # 获得所有图片的路径
    all_dirs = os.listdir(path)
    # 过滤获取所有图片的,获得符合要求的图片名字不是全路径
    # image_dirs = [i for i in all_dirs if i.endswith(".jpg") or i.endswith(".jpeg") or i.endswith(".png")]
    # 下载下来的图片名字不太规则去除过滤逻辑
    image_dirs = [i for i in all_dirs]
    # 图片数目
    number_of_images = len(image_dirs)
    images = []
    print("{} images are being loaded...".format(data_set[:-1]))
    for c, i in enumerate(image_dirs):
        # 裁切 平滑 所有像素RGB /127.5 -1.0 进行归一化 (255/27.5-1.0 =1)
        images.append(
            np.array(ImageOps.fit(Image.open(path + '/' + i), (image_size, image_size), Image.ANTIALIAS)) / 127.5 - 1.0)
        sys.stdout.write("\r Loading : {}/{}".format(c + 1, number_of_images))
    print("\n")
    # -1 表示的是依据实际情况定的 这里 -1其实是图片的数目即 number_of_images ,因为 后三维 本身就是 64 * 64 * 3,这就是原先图片的样本大小
    images = np.reshape(images, [-1, image_size, image_size, 3])
    return images.astype(np.float32)


def conv2d(x, inputFeatures, outputFeatures, name):
    """
    实现卷积层的函数
    
    :param x: 
    :param inputFeatures: 
    :param outputFeatures: 
    :param name: 
    :return: 
    """
    """
    这里说明一下 tensorflow 变量的两种创建方式
    tf.get_variable() 以及 tf.Variable() 是 TensorFlow 中创建变量的两种主要方式；
    如果在 tf.name_scope() 环境下分别使用 tf.get_variable() 和 tf.Variable()，
    两者的主要区别在于 
    tf.get_variable() 创建的变量名不受 name_scope 的影响；
    tf.get_variable() 创建的变量，name 属性值不可以相同；tf.Variable() 创建变量时，name 属性值允许重复（底层实现时，会自动引入别名机制）
    此外 tf.get_variable() 与 tf.Variable() 相比，多了一个 initilizer （初始化子）可选参数； 
    tf.Variable() 对应地多了一个 initial_value 关键字参数，也即对于 tf.Variable 创建变量的方式，必须显式初始化；
    """
    with tf.variable_scope(name):
        w = tf.get_variable("w", [5, 5, inputFeatures, outputFeatures],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b = tf.get_variable("b", [outputFeatures], initializer=tf.constant_initializer(0.0))
        # http://code.replays.net/201705/79898.html 卷积介绍  tf.nn.conv2d TensorFlow 里的卷积方法
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME") + b

        return conv


def conv_transpose(x, outputShape, name):
    """
    实现卷积转置的函数
    :param x: 
    :param outputShape: 
    :param name: 
    :return: 
    """
    with tf.variable_scope(name):
        w = tf.get_variable("w", [5, 5, outputShape[-1], x.get_shape()[-1]],

                            initializer=tf.truncated_normal_initializer(stddev=0.02))

        b = tf.get_variable("b", [outputShape[-1]], initializer=tf.constant_initializer(0.0))

        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1, 2, 2, 1])

        return convt


# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    """
    实现致密完全连接层的函数
    :param x: 
    :param inputFeatures: 
    :param outputFeatures: 
    :param scope: 
    :param with_w: 
    :return: 
    
    """
    with tf.variable_scope(scope or "Linear"):

        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32,

                                  tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU function
    :param x: 
    :param leak: 
    :param name: 
    :return: 
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator(z, z_dim):
    """
    Used to generate fake images to fool the discriminator.
    用上图中的体系架构构建一个生成器。诸如除去所有完全连接层，仅在发生器上使用ReLU以及使用批量归一化，这些任务DCGAN要求已经达标

    :param z: The input random noise.

    :param z_dim: The dimension of the input noise.

    :return: Fake images -> [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]

    """
    gf_dim = 64
    z2 = dense(z, z_dim, gf_dim * 8 * 4 * 4, scope='g_h0_lin')
    h0 = tf.nn.relu(tf.contrib.layers.batch_norm.batch_norm(tf.reshape(z2, [-1, 4, 4, gf_dim * 8]),
                               center=True, scale=True, is_training=True, scope='g_bn1'))
    h1 = tf.nn.relu(tf.contrib.layers.batch_norm.batch_norm(conv_transpose(h0, [mc.BATCH_SIZE, 8, 8, gf_dim * 4], "g_h1"),
                               center=True, scale=True, is_training=True, scope='g_bn2'))
    h2 = tf.nn.relu(tf.contrib.layers.batch_norm.batch_norm(conv_transpose(h1, [mc.BATCH_SIZE, 16, 16, gf_dim * 2], "g_h2"),
                               center=True, scale=True, is_training=True, scope='g_bn3'))
    h3 = tf.nn.relu(tf.contrib.layers.batch_norm.batch_norm(conv_transpose(h2, [mc.BATCH_SIZE, 32, 32, gf_dim * 1], "g_h3"),
                               center=True, scale=True, is_training=True, scope='g_bn4'))

    h4 = conv_transpose(h3, [mc.BATCH_SIZE, 64, 64, 3], "g_h4")

    return tf.nn.tanh(h4)


def discriminator(image, reuse=False):
    """
    我们再次避免了密集的完全连接的层，使用了Leaky ReLU，并在Discriminator处进行了批处理
    Used to distinguish between real and fake images.

    :param image: Images feed to the discriminate.

    :param reuse: Set this to True to allow the weights to be reused.

    :return: A logits value.

    """
    df_dim = 64
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h0 = lrelu(conv2d(image, 3, df_dim, name='d_h0_conv'))
    h1 = lrelu(batch_norm(conv2d(h0, df_dim, df_dim * 2, name='d_h1_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn1'))
    h2 = lrelu(batch_norm(conv2d(h1, df_dim * 2, df_dim * 4, name='d_h2_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn2'))
    h3 = lrelu(batch_norm(conv2d(h2, df_dim * 4, df_dim * 8, name='d_h3_conv'),
                          center=True, scale=True, is_training=True, scope='d_bn3'))
    h4 = dense(tf.reshape(h3, [-1, 4 * 4 * df_dim * 8]), 4 * 4 * df_dim * 8, 1, scope='d_h3_lin')
    return h4


G = generator(zin, z_dim)  # G(z)

Dx = discriminator(images)  # D(x)

Dg = discriminator(G, reuse=True)  # D(G(x))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, targets=tf.ones_like(Dx)))

d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.zeros_like(Dg))) d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.zeros_like(Dg)))

dloss = d_loss_real + d_loss_fake

gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.ones_like(Dg)))

# Get the variables which need to be trained

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]

g_vars = [var for var in t_vars if 'g_' in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(dloss, var_list=d_vars)

g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(gloss, var_list=g_vars)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

    if not load:

        for idx in range(n_iter):

            batch_images = next_batch(real_img, batch_size=batch_size)

            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

            for k in range(1):
                sess.run([d_optim], feed_dict={images: batch_images, zin: batch_z})

            for k in range(1):
                sess.run([g_optim], feed_dict={zin: batch_z})

            print("[%4d/%4d] time: %4.4f, " % (idx, n_iter, time.time() - start_time))

            if idx % 10 == 0:
                # Display the loss and run tf summaries

                summary = sess.run(summary_op, feed_dict={images: batch_images, zin: batch_z})

                writer.add_summary(summary, global_step=idx)

                d_loss = d_loss_fake.eval({zin: display_z, images: batch_images})

                g_loss = gloss.eval({zin: batch_z})

                print("\n Discriminator loss: {0} \n Generator loss: {1} \n".format(d_loss, g_loss))

            if idx % 1000 == 0:
                # Save the model after every 1000 iternations

                saver.save(sess, saved_models_path + "/train", global_step=idx)


