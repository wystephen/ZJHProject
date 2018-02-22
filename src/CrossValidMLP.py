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
from torch.utils.data import *

from sklearn import metrics

import numpy as np

import scipy as sp
import matplotlib.pyplot as plt

from visdom import Visdom

from src import DataLoder

from src import visualize
import time

from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # vis = Visdom()
    # vis.text('hello word!')
    # visual = visualize.Visualizer(env='main' + time.asctime(time.localtime(time.time())))
    tensor_board_writer = SummaryWriter()
    time_stamp_str = time.asctime(time.localtime(time.time()))

    dl = DataLoder.ZJHDataset()
    train_x, train_y, valid_x, valid_y, test_x, test_y = dl.getTrainValidTest(0.6, 0.2, 0.00002)

    print(train_x.shape, train_y.shape,
          valid_x.shape, valid_y.shape,
          test_x.shape, test_y.shape)
    t_batch_size = 1000  # train_x.shape[0]

    train_loader = DataLoader(TensorDataset(data_tensor=FloatTensor(train_x),
                                            target_tensor=FloatTensor(train_y.reshape([-1, 1]))),
                              batch_size=t_batch_size,
                              shuffle=True,
                              num_workers=4)
    train_x = Variable(FloatTensor(train_x)).cuda()
    train_y = Variable(FloatTensor(train_y.reshape([-1, 1]))).cuda()
    train_y_cpu = train_y.cpu().data.numpy()

    valid_x = Variable(FloatTensor(valid_x)).cuda()
    valid_y = Variable(FloatTensor(valid_y.reshape([-1, 1]))).cuda()
    valid_y_cpu = valid_y.cpu().data.numpy()
    test_x = Variable(FloatTensor(test_x)).cuda()
    test_y = Variable(FloatTensor(test_y.reshape([-1, 1]))).cuda()

    model = torch.nn.Sequential(torch.nn.Linear(8, 80),
                                torch.nn.RReLU(),
                                torch.nn.BatchNorm1d(80),
                                torch.nn.Linear(80, 80),
                                torch.nn.RReLU(),
                                torch.nn.BatchNorm1d(80),
                                torch.nn.Linear(80, 40),
                                # torch.nn.Dropout(0.8),
                                torch.nn.RReLU(),
                                torch.nn.BatchNorm1d(40),
                                torch.nn.Linear(40, 20),
                                torch.nn.RReLU(),
                                torch.nn.BatchNorm1d(20),
                                torch.nn.Linear(20, 40),
                                torch.nn.RReLU(),
                                torch.nn.Linear(40, 40),
                                #orch.nn.Dropout(0.8),
                                torch.nn.RReLU(),
                                torch.nn.BatchNorm1d(40),
                                torch.nn.Linear(40, 20),
                                torch.nn.Dropout(0.8),
                                torch.nn.RReLU(),
                                torch.nn.Linear(20, 10),
                                torch.nn.RReLU(),
                                torch.nn.Linear(10, 1)
                                )

    loss_fn = torch.nn.MSELoss(size_average=False)

    model.cuda()
    loss_fn.cuda()
    loss_array = np.zeros(1000000)
    # optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=0.00001,
    #                             momentum=0.05)
    optimizer = torch.optim.RMSprop(model.parameters())
    # optimizer = torch.optim.Adadelta(model.parameters())
    max_train_r2 = 0.0
    max_valid_r2 = 0.0
    min_train_loss = 100000000.0
    min_valid_loss = 100000000.0

    for epoch in range(10000):
        model.train()
        for i_batch, sample_batched in enumerate(train_loader):
            # print(i_batch)
            b_x, b_y = sample_batched
            b_x = Variable(b_x).cuda()
            b_y = Variable(b_y, requires_grad=False).cuda()
            # print(b_x.shape,b_y.shape)
            y_pre = model(b_x)
            loss = loss_fn(y_pre, b_y)
            # print(epoch, i_batch, loss.data[0])
            # metrics.r2_score(b_y.cpu().data.numpy(),
            #                  y_pre.cpu().data.numpy()))
            model.zero_grad()
            loss.backward()

            optimizer.step()

        model.eval()
        train_r2 = metrics.r2_score(train_y_cpu,
                                    model(train_x).cpu().data.numpy())
        train_loss = loss_fn(model(train_x), train_y).data[0]

        valid_r2 = metrics.r2_score(valid_y_cpu,
                                    model(valid_x).cpu().data.numpy())
        valid_loss = loss_fn(model(valid_x), valid_y).data[0]

        # test_r2 = metrics.r2_score(test_y.cpu().data.numpy(),
        #                            model(test_x).cpu().data.numpy())
        # visual.plot('train_r2', train_r2)
        # visual.plot('valid_r2', valid_r2)
        # visual.plot('test_r2',test_r2)
        # visual.plot('train_loss', train_loss)
        # visual.plot('valid_loss', valid_loss)
        if (max_train_r2 < train_r2):
            torch.save(model, './local_model/' + time_stamp_str)
            model.cuda()
        max_train_r2 = max(max_train_r2, train_r2)
        max_valid_r2 = max(max_valid_r2, valid_r2)
        print(epoch, ':{', 'train r2:', train_r2,
              ',max train r2:', max_train_r2,
              ',train loss:', train_loss,
              ',valid_r2:', valid_r2,
              ',max valid r2:', max_valid_r2,
              ',valid loss:', valid_loss, '}')
        tensor_board_writer.add_scalars('data/MSE',
                                        {
                                            'trainloss': train_loss,
                                            'validloss': valid_loss},
                                        epoch
                                        )
        tensor_board_writer.add_scalars('data/r2',
                                        {
                                            'trainr2': train_r2,
                                            'validr2': valid_r2
                                        },
                                        epoch)
        for name, param in model.named_parameters():
            tensor_board_writer.add_histogram(
                name, param.clone().cpu().data.numpy(),
                epoch
            )

            model.train()
