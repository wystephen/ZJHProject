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

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import *

class ZJHDataset:
    def __init__(self, file_name='/home/steve/Data/ZJHData/source_data.csv'):
        self.file_name = file_name
        self.data = np.loadtxt(self.file_name,delimiter=',')
        self.normlized_data = normalize(self.data)
        print(self.normlized_data.mean(axis=1))

    # def __getitem__(self, index):
    #     return self.data[index]

    # def __len__(self):
    #     return self.normlized_data.shape[0]

if __name__ == '__main__':
    dl = DataLoader(file_name='/home/steve/Data/ZJHData/NO2_data.csv')

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
