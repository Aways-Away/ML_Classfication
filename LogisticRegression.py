# -*- coding : utf-8 -*-
# @Time      : 2020/11/29 19:43
# @Author    : A WAYS AWAY
# @File      : LogisticRegression.py
# @IDE       : PyCharm

import numpy as np
import datetime
import matplotlib.pyplot as plt

starttime = datetime.datetime.now()
np.random.seed(0)  # 每次生成的随机数相同
# print(np.random.rand(4))

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'  # 用于测试集的预测输出

# 加载数据到 numpy array
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


# 正则化训练集和测试集
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_mean, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

# 将训练集分为测试集和测试集
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


# ---------------- _shuffle, _sigmoid, _f, _predict, _accuracy函数的定义 ---------------- #
def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


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


# ---------------- 梯度及损失函数的定义 ---------------- #
def _cross_entropy_loss(y_pred, Y_label):
    '''
    :param y_pred: 概率预测，浮点向量
    :param Y_label: ground truth labels, bool vector
    :return: 交叉熵，标量
    '''
    # Loss^n = -[y^n * ln(p^n) + (1-y^n) * ln(1-p^n)] #
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy


def _gradient(X, Y_label, w, b):
    # 计算权重w和偏差b的交叉熵损失的梯度
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, axis=1)  # 求pred_error * X转置矩阵, 每一行的和
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


# ---------------- 训练 ---------------- #
# 初始化权重w和偏差b
w = np.zeros((data_dim,))
b = np.zeros((1,))

max_iter = 10
batch_size = 8
learning_rate = 0.2

# 保留每次迭代的损失和准确性并绘图
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

step = 1  # 计算参数更新次数

for epoch in range(max_iter):  # train
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini_batch training
    for idx in range(int(np.floor((train_size / batch_size)))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)

        # 更新权重矩阵
        # learning rate decay with time
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1
    # compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
    print('after {} epoches, the (acc, loss) on train data is:'.format(epoch),_accuracy(Y_train_pred, Y_train), _cross_entropy_loss(y_train_pred, Y_train)/train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / train_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

# ---------------- 绘制accuracy和loss curve ---------------- #
# accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png', dpi=600)
# plt.show()

# loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png', dpi=600)
# plt.show()

# ---------------- 测试集上的预测 ---------------- #
predictions = _predict(X_test, w, b)
print('X_test.shape:{}, Y_train:{}', X_test.shape, Y_train.shape)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
# print('The accuracy on test set is: {}'.format(_accuracy(np.round(predictions), Y_train)))


# ---------------- 最重要的权重 ---------------- #
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('.\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])
print(len(content))


# ----------------程序运行时间---------------- #
endtime = datetime.datetime.now()
runtime = (endtime - starttime).seconds
print('运行时间: {}s '.format(runtime))

