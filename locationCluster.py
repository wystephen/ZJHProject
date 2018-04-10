# -*- coding:utf-8 -*-
# Created by steve @ 18-2-28 下午9:42
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

import numpy  as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.spatial.distance import  pdist

from src import DataLoder
from sklearn.cluster import KMeans

if __name__ == '__main__':
    dl = DataLoder.ZJHDataset()
    pose_data = dl.data[:,-3:-1]
    value_data = dl.data[:,0]
    # print(pose_data)
    time_data = dl.data[:,-1]

    plt.figure()
    plt.plot(pose_data[:,0],pose_data[:,1],'*r')
    plt.show()

    # plt.figure()
    # plt.plot(time_data,'*-r')
    # plt.show()
    n_cluster = 10
    k_means = KMeans()


