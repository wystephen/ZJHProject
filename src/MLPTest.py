# -*- coding:utf-8 -*-
# Created by steve @ 18-2-4 下午11:54
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

import numpy as np

import scipy as sp
import matplotlib.pyplot as plt

from src import DataLoder

if __name__ == '__main__':
    dl = DataLoder.DataLoader()
    all_x_data = dl.normlized_data[:, 1:]
    all_y_data = dl.normlized_data[:, 0]

    print(all_x_data.shape)
    model = torch.nn.Sequential(torch.nn.Linear(9,20),
                                torch.nn.ReLU(),
                                torch.nn.Linear(20,10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10,1),
                                )
    loss_fn = torch.nn.MSELoss(size_average=False)

    x = Variable(FloatTensor(all_x_data).cuda())
    y = Variable(FloatTensor(all_y_data.reshape(-1,1)).cuda())
    x.cuda()
    y.cuda()

    model.cuda()
    loss_fn.cuda()

    learning_rate = 1e-4
    for t in range(1000):
        y_pred = model(x)
        print(y_pred.data.shape)
        print(y.data.shape)

        loss = loss_fn(y,y_pred)
        print(t,loss.data[0])
        model.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.data -= learning_rate*param.grad.data

