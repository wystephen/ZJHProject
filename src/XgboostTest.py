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
from sklearn.model_selection import train_test_split


from src import DataLoder

import math

def TenFoldTrain(data,max_depth=8,trees = 200):
    all_score = list()
    pre_score = list()
    vaild_score = list()
    kf = sk.model_selection.KFold(n_splits=10)
    for train,test in kf.split(data):
        reg = xgb.XGBRegressor(max_depth=max_depth,learning_rate=0.1,n_estimators=trees,nthread=10)
        reg.fit(data[train,1:],data[train,0].reshape([-1,1]))
        all_score.append(sk.metrics.r2_score(data[:,0].reshape([-1,1]),reg.predict(data[:,1:])))
        pre_score.append(sk.metrics.r2_score(data[train,0].reshape([-1,1]),reg.predict(data[train,1:])))
        vaild_score.append(sk.metrics.r2_score(data[test,0].reshape([-1,1]),reg.predict(data[test,1:])))
    print(all_score,pre_score,vaild_score)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == '__main__':
     dl = DataLoder.DataLoader('/home/steve/Data/ZJHData/NO2_data.csv')

     x = dl.data[:,1:]
     y = dl.data[:,0].reshape([-1,1])
     #
     # x_train, x_vaild,y_train, y_valid = train_test_split(x,y,test_size=0.3)
     #
     reg = xgb.XGBRegressor()
     #
     # reg.fit(x_train,y_train)
     #
     # print('all r^2:',sk.metrics.r2_score(y,reg.predict(x)))
     # print('pre r^2:',sk.metrics.r2_score(y_train,reg.predict(x_train)))
     # print('val r^2:',sk.metrics.r2_score(y_valid,reg.predict(x_vaild)))
     # TenFoldTrain(dl.data,5,100)

     para_dist = {
         "max_depth":[2,5,4,6,8,10,14,18,20,24],
         "learning_rate":[0.1,0.2,0.3,0.05,0.01,0.007,0.002],
         "booster":["gbtree","gblinear","dart"],
         "n_estimators":[10,40,78,100,130,170,200],
         "n_jobs":[6]
     }


     # random_search = sk.model_selection.RandomizedSearchCV(reg,param_distributions=para_dist,
     #                                                       scoring=sk.metrics.make_scorer(sk.metrics.r2_score))

     random_search = sk.model_selection.RandomizedSearchCV(reg,param_distributions=para_dist,
                                                           n_iter=100,
                                                           scoring=sk.metrics.make_scorer(sk.metrics.r2_score))
                                                           # scoring=sk.metrics.r2_score)
     random_search.fit(x,y)
     report(random_search.cv_results_,n_top=10)


