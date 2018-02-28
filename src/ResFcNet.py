# -*- coding:utf-8 -*-
# Created by steve @ 18-2-28 下午11:21
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
from torch import  nn


class ResBlock(nn.Module):

    def __init__(self,in_size):
        super(ResBlock, self).__init__()

        self.fc1 = nn.Linear(in_size,in_size)
        self.bn1 = nn.BatchNorm1d(in_size)
        self.relu = nn.RReLU(inplace=True)

        self.fc2 = nn.Linear(in_size,in_size)
        self.bn2 = nn.BatchNorm1d(in_size)
        # self.relu2 = nn.RReLU(inplace=True)


    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.fc1(x)))
        out = self.relu(self.bn2(self.fc2(out)))

        out += residual
        return out










class ResFcNet(object):



    def __init_(self,input_size,output_size,Layer_size=[10,10]):
        self.layer_list = list()

