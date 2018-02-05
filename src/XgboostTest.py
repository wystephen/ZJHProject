# -*- coding:utf-8 -*-
# Created by steve @ 18-2-5 下午3:54
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

import sklearn as sk
import xgboost as xgb


from src import DataLoder

if __name__ == '__main__':
     dl = DataLoder.DataLoader('/home/steve/Data/ZJHData/NO2_data.csv')

     x = dl.data[:,1:]
     y = dl.data[:,0].reshape([-1,1])

     reg = xgb.XGBRegressor()

     reg.fig(x,y)

     print(sk.metrics.r2_score(y,reg.predict(x)))


