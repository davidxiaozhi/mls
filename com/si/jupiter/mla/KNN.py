# -*- coding:UTF-8 -*-
"""
    K 近邻算法复习,其为一种类别判定算法,原理是计算样本 A 的 K 个相近的邻居,这些邻居中大多数归属于那个分类,该样本 A 便属于那个分类,
    要防止K 过少时 过拟合的问题,过拟合可以看知乎上的一篇文章特逗
    https://www.zhihu.com/question/32246256
    优点：精度高，对异常值不敏感
    缺点：计算复杂度高，空间复杂度高
    
    分类器的错误率:
    分类器给出错误结果的次数除以测试执行的总数。
    错误率是常用的评估方法,主要用于评估分类器在某个数据集上的执行效果。完美分类器的错误率为 0,
    最差分类器的错误率是 1.0 ,在这种情况下,分类器根本就无法找到一个正确答案
    
    这里说一下numpy 新版本得使用np.array
    tile函数
        from numpy import *
        a=[2,1]
        b=tile(a,(3,2)) 元组第一个参数代表的行 第二个参数代表的是列
        b
        array([[2, 1, 2, 1],
               [2, 1, 2, 1],
               [2, 1, 2, 1]])
        c=tile(a,3)
        c
        array([2, 1, 2, 1, 2, 1])
        
    
"""

# from numpy import *
import numpy as np


def createDataSet():
    # 每行都是一个样本
    group = np.array([[1.0, 1.1],
                      [1.0, 1.0],
                      [0, 0],
                      [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataSet, labels, k):
    """
    :param intX: 输入的分类向量
    :param dataSet: 输入的训练样本集
    :param label: 标签向量label ，标签项目的数目与矩阵dataSet的行数相同
    :param k:     邻居数目选择
    :return: 
    """
    dataSetSize = dataSet.shape[0]  # 二维数组有多少行 其实就是 样本的数目

    # ---下面开始距离计算----
    # 这里介绍一下tile函数 第一个参数A代表的是初始值向量 第二参数B可以是一个数值类型 这样代表 A 在 列方向重复B次、
    # 如果B是元组，一般二维元组 那么表明在B1 代表行上重复的次数 B2代表的是在列上重复的次数
    # 下面欧几里得描述都按一行即一个目标样本描述
    diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet  # 每个样本 做欧几里得的第一步 点与点之间做差 [(x1-x'1),(y1-y'1)]
    sqDiffMat = diffMat**2               # 对上一步的差求平方   [(x1-x'1)**2,(y1-y'1)**2]
    sqDistances = sqDiffMat.sum(axis=1)  # 对上一步求和 [(x1-x'1)**2+(y1-y'1)**2] 多维度变一维度
    distances = sqDistances**0.5         # 对上一步开平方( (x1-x'1)**2 + (y1-y'1)**2 )**0.5
    sortedDistIndicies = distances.argsort()

    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        votellable = labels[sortedDistIndicies[i]]
        classCount[votellable] = classCount.get(votellable,0)+1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[0], reverse=True)
    return sortedClassCount[0][0]


def main():
    group, labels = createDataSet()
    result = classify0([0.0],group,labels,3)
    print(result)


if __name__ == '__main__':
    main()






