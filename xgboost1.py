# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:30:24 2018

@author: Administrator
随机森林不需要预处理数据
"""
import xgboost as xgb
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
#中文字体设置
from matplotlib.font_manager import FontProperties
font=FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14)

#读取变量名文件
varibleFileName="titantic.xlsx"
#读取目标文件
targetFileName="target.xlsx"
#读取excel
data=pd.read_excel(varibleFileName)
data_dummies=pd.get_dummies(data)
print('features after one-hot encoding:\n',list(data_dummies.columns))
features=data_dummies.ix[:,"Pclass":'Embarked_S']
x=features.values

#数据预处理
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0) 
imp.fit(x)
x=imp.transform(x)


target=pd.read_excel(targetFileName)
y=target.values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
names=features.columns

dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)

params={'booster':'gbtree',
    #'objective': 'reg:linear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}


watchlist = [(dtrain,'train')]

bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

ypred=bst.predict(dtest)

# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1

#模型校验
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)

print("xgboost:")  
print('Feature importances:{}'.format(bst.get_fscore()))

'''
AUC: 0.9464
ACC: 0.8841
Recall: 0.8716
F1-score: 0.8716
Precesion: 0.8716
xgboost:
Feature importances:{'f5': 69, 'f1': 178, 'f2': 68, 'f4': 245, 'f6': 25, 'f0': 88, 'f3': 25, 'f194': 4, 'f193': 21, 'f195': 9}
'''
