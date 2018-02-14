# -*- coding:utf-8 -*-
# Created by steve @ 18-2-14 下午2:27
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
import torch
from torch.autograd import Variable
from torch import FloatTensor
from torch.utils.data import Dataset,DataLoader

from sklearn import metrics

import numpy as np

import scipy as sp
import matplotlib.pyplot as plt

from visdom import Visdom

from src import DataLoder

if __name__ == '__main__':

    vis = Visdom()
    vis.text('hello word!')

    dl = DataLoder.ZJHDataset()
    train_x, train_y,valid_x,valid_y,test_x,test_y = dl.getTrainValidTest()




