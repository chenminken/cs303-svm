'''
author：ken
references:
[1]svmv2.pdf of sustc CS303 class.
[2]smo.pdf of standford CS229 class。
'''

import getopt
import sys
import time

import numpy as np
#todo command line reading

#todo data reading


class SVMv2:
    def __init__(self,x,y,epochs=10000,learning_rate=0.01):
        # 加一个维度。x是二维矩阵，使得w0+w1*x1+......可行。
        self.x = np.c_[np.ones((x.shape[0])),x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1],)


    # 需要改变目标函数。
    def get_loss(self,x,y):
        loss = max(0, 1-y* np.dot(x,self.w))
        return loss

    # 不能完全用线性可分的代码来写
    def cal_sgd(self, x, y, w):
        if y * np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        return w

    def train(self, timeBudget):

        avg_time_taken = 0
        acm_time_taken = 0
        time_left = timeBudget
        count = 0
        for epoch in range(self.epochs):
            time_start = time.time()
            if avg_time_taken > 2 * time_left or time_left < 1:
                break
            randomize = np.arange(len(self.x))
            # 打乱数组顺序
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi,self.w)

            count +=1
            time_end = time.time()
            time_taken = time_end - time_start
            time_left -= time_taken
            acm_time_taken += time_taken
            avg_time_taken = acm_time_taken / (epoch + 1)
            # print('ephco: {0} loss: {1}'.format(epoch, loss))
        # print(count)

    def predict(self,x):
        x_test = np.c_[np.ones((x.shape[0])), x]
        return np.sign(np.dot(x_test, self.w))

if __name__ == '__main__':
    start_time = time.time()
    traindata = ''
    testdata = ''
    timemax = 60
    traindata = sys.argv[1]
    testdata = sys.argv[2]
    if len(sys.argv) == 5:
        timemax = sys.argv[4]
    traindata = np.loadtxt(traindata)
    testdata = np.loadtxt(testdata)
    train_data = traindata[:,0:-1]
    train_label = traindata[:,-1].astype(int)
    test_data = testdata[:,0:-1]
    # test_label = testdata[:,-1].astype(int)

    # svm
    timemax = 10
    svmv2 = SVMv2(train_data,train_label)
    svmv2.train(timemax)
    predict_label = svmv2.predict(test_data)
    count = 0
    test_size = test_data.shape[0]
    for i in range(test_size):
        print(int(predict_label[i]))
    # for i in range(test_size):
    #     if int(test_label[i]) == int(predict_label[i]):
    #         count += 1
    # print("{}/{}={}".format(str(count),str(test_size),str(count/test_size)))
    # print(svmv2.w)
    # print(time.time()-start_time)
