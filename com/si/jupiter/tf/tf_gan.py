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
    images = np.reshape(images, [-1, image_size, image_size, 3])
    return images.astype(np.float32)
