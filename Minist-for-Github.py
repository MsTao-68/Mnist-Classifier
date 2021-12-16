# !usr/bin/python
# encoding: utf-8
# Author: Tracy Tao (Dasein)
# Date: 2021/11/31

import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version = 1, cache=True) #也可以不用手动下载数据集， 直接用现成包里的数据
mnist.target = mnist.target.astype(np.int8)
from sklearn.model_selection import train_test_split #数据集训练集划分
from sklearn import metrics # 模型评分
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans # K平均聚类模型
from sklearn.ensemble import RandomForestClassifier # 随机森林模型
import time

def readfile(file1, file2):     # 读取txt文件
    '''
    :param file1:
    :param file2:
    :return:
    '''
    binFile = open(file1,'rb')      # 读取图像数据文件
    binFile_buf = binFile.read()

    Label = open(file2,'rb')        # 读取label文件
    lbl = Label.read()
    return binFile_buf, lbl

def get_image(binFile_buf):     # 图像数据二进制转码
    image_idx = 0
    image_idx += struct.calcsize('>IIII')
    magic, nImage, nImgRows, nImgCols = struct.unpack_from('>IIII', binFile_buf, 0)
    im = []
    for i in range(100):
        tmp = struct.unpack_from('>784B', binFile_buf, image_idx)
        im.append(np.reshape(tmp, (28,28)))
        image_idx += struct.calcsize('>784B')
    return im

def get_label(lbl):     # Label数据二进制转码
    label_idx = 0
    label_idx += struct.calcsize('>II')
    return struct.unpack_from('>100B', lbl, label_idx)


if __name__ == "__main__":
    test_image_data, test_label_data = readfile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte') # 读取raw data txt
    test_im = get_image(test_image_data) # 获取图像数据
    # print(test_im)
    test_label = get_label(test_label_data) # 获取label数据
    # print(test_label)
    plt.figure(figsize=(20, 20), dpi=80) # 数据可视化，看是否成功展示黑白的数字图片
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        title = str(test_label[i])
        plt.title(title)
        plt.imshow(test_im[i], cmap='gray')
    # plt.savefig('TestData.png', dpi =80)
    plt.show()

    train_image_data, train_label_data = readfile('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    train_im = get_image(train_image_data)
    train_label = get_label(train_label_data)
    plt.figure(figsize=(20, 20), dpi=80)
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        title = str(train_label[i])
        plt.title(title)
        plt.imshow(train_im[i], cmap='gray')
    # plt.savefig('TrainData.png', dpi =80)
    plt.show()

    # Start From Here
    print(mnist.data.shape)
    print(mnist.target.shape) # 查看数据量
    X, y = mnist.data, mnist.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=22) # 划分数据集

    # 随机梯度下降模型
    sgd_clf = SGDClassifier(max_iter=5, tol=- np.infty, random_state=42)
    sgd_clf.fit(x_train, y_train)
    y_pred = sgd_clf.predict(x_test)
    print(metrics.adjusted_rand_score(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("随机梯度下降模型准确率：", accuracy)

    # Kmeans模型
    kmean = KMeans(n_clusters=10)   # 0-9 分成10类
    kmean.fit(x_train, y_train)
    kmean_pred = kmean.predict(x_test)
    kmean_accru = metrics.adjusted_rand_score(y_test, kmean_pred)
    print("Kmeans模型准确率：", kmean_accru)

    # 随机森林模型
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    rf_accru = metrics.adjusted_rand_score(y_test, rf_pred)
    print("随机森林模型准确率：", rf_accru)

    # 模型汇总
    Classifiers = [["Random Forest", RandomForestClassifier()],
                   ['KNN', KNeighborsClassifier(n_neighbors=10)],
                   ['KMeans', KMeans(n_clusters=10)],
                   ['SGDClassifier', SGDClassifier(max_iter=5, tol=- np.infty, random_state=42)]]
    Classify_result = []
    names = []
    prediction = []
    for name, classifier in Classifiers:
        classifier = classifier
        tic = time.time()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)
        toc = time.time()
        time_diff = toc - tic # 模仿Andrew Ng Deep Learning的视频里的写法：tictoc LOL :p
        accuracy = accuracy_score(y_test, y_pred)
        score = adjusted_rand_score(y_test, y_pred)
        class_eva = pd.DataFrame([accuracy, score, time_diff])
        Classify_result.append(class_eva)
        name = pd.Series(name)
        names.append(name)
        y_pred = pd.Series(y_pred)
        prediction.append(y_pred)
    names = pd.DataFrame(names)
    names = names[0].tolist()
    result = pd.concat(Classify_result, axis=1)
    result.columns = names
    result.index = ['accuracy', 'adjusted_rand_score', 'time_diff']
    print("Result:", result)





