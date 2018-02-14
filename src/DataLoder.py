# -*- coding:utf-8 -*-
# Created by steve @ 18-2-4 下午10:13
'''
                   _ooOoo_ 
                  o8888888o 
                  88" . "88 
                  (| -_- |) 
                  O\  =  /O 
               ____/`---'\____ 
             .'  \\|     |//  `. 
            /  \\|||  :  |||//  \ 
           /  _||||| -:- |||||-  \ 
           |   | \\\  -  /// |   | 
           | \_|  ''\---/''  |   | 
           \  .-\__  `-`  ___/-. / 
         ___`. .'  /--.--\  `. . __ 
      ."" '<  `.___\_<|>_/___.'  >'"". 
     | | :  `- \`.;`\ _ /`;.`/ - ` : | | 
     \  \ `-.   \_ __\ /__ _/   .-` /  / 
======`-.____`-.___\_____/___.-`____.-'====== 
                   `=---=' 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
         佛祖保佑       永无BUG 
'''


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

# from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split

class ZJHDataset:
    def __init__(self, file_name='/home/steve/Data/ZJHData/NO2_data.csv'):
        self.file_name = file_name
        self.data = np.loadtxt(self.file_name,delimiter=',')
        self.normlized_data = normalize(self.data)
        print(self.data.mean(axis=1))
        print(self.data.std(axis=1))
        print(self.normlized_data.mean(axis=1))
        print(self.normlized_data.std(axis=1))

    # def __getitem__(self, index):
    #     return self.data[index]

    # def __len__(self):
    #     return self.normlized_data.shape[0]
    def getTrainValidTest(self,train_rate,valid_rate,test_rate):
        train_rate = float(train_rate)
        valid_rate = float(valid_rate)
        test_rate = float(test_rate)
        if(train_rate+valid_rate+test_rate)>1.0:
            sum = train_rate+valid_rate+test_rate
            train_rate = train_rate /sum
            valid_rate = valid_rate / sum
            test_rate = test_rate /sum

        tvx,test_x,tvy,test_y = train_test_split(self.data[:,1:],self.data[:,0],
                                                 shuffle=True,
                                                 test_size=test_rate
                                                 )
        train_x, valid_x, train_y,valid_y = train_test_split(
            tvx[:,1:],tvx[:,0],
            shuffle=True,
            test_size=valid_rate/(1-test_rate)
        )

        return train_x,train_y,valid_x,valid_y,test_x,test_y

if __name__ == '__main__':
    dl = ZJHDataset()
    plt.figure()
    plt.plot(dl.data[:,0])

    plt.figure()
    for i in range(1,dl.data.shape[1]):
        plt.plot(dl.data[:,i],label=str(i))
    plt.legend()
    plt.grid()
    plt.figure()
    for i in range(1,dl.data.shape[1]):
        plt.plot(dl.normlized_data[:,i],label=str(i))
    plt.legend()
    plt.grid()


    plt.show()
