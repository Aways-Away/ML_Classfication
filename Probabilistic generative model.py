# -*- coding : utf-8 -*-
# @Time      : 2020/12/2 13:58
# @Author    : A WAYS AWAY
# @File      : Probabilistic generative model.py
# @IDE       : PyCharm

import numpy as np
import datetime
import matplotlib.pyplot as plt

starttime = datetime.datetime.now()
np.random.seed(0)  # 每次生成的随机数相同

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'  # 用于测试集的预测输出

# 加载数据
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):  # 方法私有
    '''
    :param X: data to be processed
    :param train: 'True' when processing training dataw, 'False' for testing data
    :param specified_column: indexes of the columns that will be normalized. If 'None', all columns will be normalized.
    :param X_mean: mean value of training data, used when train = 'Fasle'
    :param X_std: standard deviation of training data, used when train = 'False'
    :return:
    '''
    if specified_column == None:
        specified_column = np.arange(X.shape[1])  # 返回[0,1,2, ..., X.shape[1]]   X.shape[1] 就是data的列数
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    return X, X_mean, X_std


def _train_dev_split(X, Y, dev_ratio=0.25):
    # 将训练集再分为训练集和测试集
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]  # 前train_size行，后train_size行


# 归一化
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


def _sigmoid(z):
    # 用于计算概率
    # 为了避免溢出，设置了最小(1e-8)、最大(1-(1e-8))输出值
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    '''
    :param X: input data, shape = [batch_size, data_dimension]
    :param w: weight vector, shape = [data_dimension, ]
    :param b: bias, scalar  偏差，标量
    :return:
    '''
    return _sigmoid(np.matmul(X, w) + b)  # matrix multiply


def _predict(X, w, b):
    # 返回X的每一行的真值预测
    # 通过逻辑回归函数四舍五入的结果
    return np.round(_f(X, w, b).astype(np.int))


def _accuracy(Y_pred, Y_label):
    # 计算预测的准确性
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


# ------------------均值和协方差------------------ #
# 均值
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis=0)
mean_1 = np.mean(X_train_1, axis=0)

# 协方差
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# ------------------权重矩阵和偏差矩阵------------------ #
# 计算协方差矩阵的逆
# 由于协方差矩阵几乎是奇异矩阵，因此np.linalg.inv()可能会产生较大的数值误差
# 通过SVD分解，可以高效而准确地求逆矩阵
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) \
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

# 在测试集上的accuracy
Y_train_pred = 1 - _predict(X_train, w, b)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

# ------------------ 在 X_test 上进行预测 并输出结果到 output_generative.csv ------------------ #
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format('generative'), mode='w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# 打印参数
# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])

# ----------------程序运行时间---------------- #
endtime = datetime.datetime.now()
runtime = (endtime - starttime).seconds
print('运行时间: {}s '.format(runtime))
