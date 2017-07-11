# -*- coding:UTF-8 -*-
"""
    K 近邻算法复习,其为一种类别判定算法,原理是计算样本 A 的 K 个相近的邻居,这些邻居中大多数归属于那个分类,该样本 A 便属于那个分类,
    要防止K 过少时 过拟合的问题,过拟合可以看知乎上的一篇文章特逗
    https://www.zhihu.com/question/32246256
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1],[1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

