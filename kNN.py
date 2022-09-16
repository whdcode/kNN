from numpy import *
import operator


# operator为运算符模块

def createdataset():
    """创建数据集和标签"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0.5, 0.5], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'C', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    """构造k近邻算法"""
    datasetsize = dataset.shape[0]  # 返回矩阵行数,shape(1)为列数
    # 函数tile():输入向量inX在矩阵x方向扩展到原来一倍，在y方向扩展到原来datasetsize倍（行数），
    # 作差为新数据与样本各个数据对应特征的差
    diffmat = tile(inX, (datasetsize, 1)) - dataset
    sqDiffmat = diffmat ** 2
    sqdistance = sqDiffmat.sum(axis=1)  # axis=1表示矩阵每一行求和，axis=0表示对矩阵列求和
    distance = sqdistance ** 0.5    # 开根号得到距离
    sorteddistanceindicies = distance.argsort()     # 函数argsort()使数组元素从小到达排序，随后返回对应原索引为元素的数组

    classcount = {}
    for i in range(k):
        voteIlabel = labels[sorteddistanceindicies[i]]
        classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1  # 此处是一个累计标签（A或B）数目的算法，存在则在原基础上加1，不存在赋予1
        # 函数get(key_name, 默认值)，字典中存在这个键，则返回对应键值，反之则返回默认值
        sortedclasscount = sorted(classcount.items(),
                                  key=operator.itemgetter(1), reverse=True)
        # 函数sorted():返回一个排序后的表格
        # 函数.items()将字典转换成元素为键值对元组的列表
        # key=operator.itemgetter(i),对排序对象下标为i的目标排序
        # reverse=True 从大到小，reverse=False  从小到大
        return sortedclasscount[0][0]


def file2matrix(filename):
    """文本记录的解析"""
    # with open(filename, 'r', encoding='UTF-8') as f:
    f = open(filename)
    arrayolines = f.readlines()
    numberoflines = len(arrayolines)
    returnmat = zeros((numberoflines, 3))
    classlabelvector = []
    index = 0
    dic = {'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3}
    for line in arrayolines:
        line = line.strip()
        listfromline = line.split('\t')
        returnmat[index, :] = listfromline[0: 3]
        classlabelvector.append(dic[listfromline[-1]])  # int明确告诉解释器listfromline[-1]为数值整型而非字符串
        index += 1

    return returnmat, classlabelvector

