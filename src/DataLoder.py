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

class DataLoader:
    def __init__(self, file_name='/home/steve/Data/ZJHData/source_data.csv'):
        self.file_name = file_name
        self.data = np.loadtxt(self.file_name,delimiter=',')
        self.normlized_data = self.data-self.data.mean(axis=0)
        self.normlized_data /= self.normlized_data.std(axis=0)

if __name__ == '__main__':
    dl = DataLoader()

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
