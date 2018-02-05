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

from sklearn import metrics

import numpy as np

import scipy as sp
import matplotlib.pyplot as plt

from src import DataLoder

if __name__ == '__main__':
    dl = DataLoder.DataLoader()
    all_x_data = dl.normlized_data[:, 1:]
    all_y_data = dl.normlized_data[:, 0]
    all_y_data = all_y_data.reshape([-1, 1])

    print(all_x_data.shape)
    model = torch.nn.Sequential(torch.nn.Linear(9, 20),
                                torch.nn.ReLU(),
                                torch.nn.Linear(20, 10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10,10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10,10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10,10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10, 1),
                                )
    loss_fn = torch.nn.MSELoss(size_average=False)

    x_torch = FloatTensor(all_x_data)
    y_torch = FloatTensor(all_y_data)

    x = Variable(FloatTensor(all_x_data).cuda())
    y = Variable(FloatTensor(all_y_data.reshape(-1, 1)).cuda(),
                 requires_grad=False)
    x.cuda()
    y.cuda()

    model.cuda()
    loss_fn.cuda()


    loss_array = np.zeros(1000000)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.ASGD(model.parameters(),lr=1e-6)
    # optimizer = torch.optim.Adadelta(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(),lr=1e-6,momentum=0.1)
    # optimizer.cuda()

    for t in range(loss_array.shape[0]):
        y_pred = model(x)
        # print(y_pred.data.shape)
        # print(y.data.shape)

        loss = loss_fn(y_pred, y)
        print(t, loss.data[0],metrics.r2_score(all_y_data,y_pred.cpu().data.numpy()))
        loss_array[t] = loss.data[0]
        model.zero_grad()
        loss.backward()
        # if(learning_rate>1e-6):
        #     learning_rate *= 0.95
        # for param in model.parameters():
        #     param.data -= learning_rate * param.grad.data
        optimizer.step()

    plt.figure()
    plt.plot(loss_array[:], '-*')
    plt.grid()
    plt.show()
