import sys
import time
import numpy as np
from SVM import SVMv2

traindata = ''
testdata = ''
# timemax = 60
raw_file = sys.argv[1]

raw_data = np.loadtxt(raw_file)
randomize = np.arange(len(raw_data))
# 打乱数组顺序
np.random.shuffle(randomize)
raw_data = raw_data[randomize]
num = int(0.7 * np.shape(raw_data)[0])
train_data = raw_data[:num, 0:-1]
train_label = raw_data[:num, -1].astype(int)
test_data = raw_data[num:, 0:-1]
test_label = raw_data[num:, -1].astype(int)
# svm
for t in range(26,50,3):
    # raw_data = np.loadtxt(raw_file)
    # randomize = np.arange(len(raw_data))
    # # 打乱数组顺序
    # np.random.shuffle(randomize)
    # raw_data = raw_data[randomize]
    # num = int(0.7 * np.shape(raw_data)[0])
    # train_data = raw_data[:num, 0:-1]
    # train_label = raw_data[:num, -1].astype(int)
    # test_data = raw_data[num:, 0:-1]
    # test_label = raw_data[num:, -1].astype(int)
    start_time = time.time()
    svmv2 = SVMv2(train_data, train_label)
    svmv2.train(t)
    predict_label = svmv2.predict(test_data)
    count = 0
    test_size = test_data.shape[0]
    for i in range(test_size):
        if int(test_label[i]) == int(predict_label[i]):
            count += 1
    print("{4}\t{2}= {1}/{0}   {3}".format(str(count), str(test_size), str(count / test_size), str(time.time() - start_time),str(t)))
    # print(svmv2.w)