"""测试k近临算法"""
import kNN
import importlib
import matplotlib
import matplotlib.pyplot as plt
from numpy import array

matplotlib.rc("font", family='Microsoft YaHei')  # 使得标签可以显示中文

group, labels = kNN.createdataset()
print(group)
Y = kNN.classify0([0.5, 0.6], group, labels, 3)
print(Y)

importlib.reload(kNN)
datingDatamat, datinglabel = kNN.file2matrix('datingTestSet.txt')
print(datinglabel[0:21])  # 'largeDoses': 1, 'smallDoses': 2, 'didntLike': 3
print(datingDatamat)

# 使用散点图绘制初始数据
plt.style.use('bmh')

fig, orin_data = plt.subplots(figsize=(15, 9))
orin_data.scatter(datingDatamat[:, 1], datingDatamat[:, 2], 15.0*array(datinglabel),
                  15.0*array(datinglabel))
orin_data.set_title("原始数据散点图", fontsize=25)
orin_data.set_xlabel("玩视频游戏所耗时间百分比", fontsize=24)
orin_data.set_ylabel("每周消耗的冰淇淋公升数", fontsize=24)
plt.savefig("原始数据散点图", bbox_inches='tight')