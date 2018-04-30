# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:30:24 2018

@author: Administrator
随机森林不需要预处理数据
"""
from sklearn.svm import SVC
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

svm=SVC()
svm.fit(x_train,y_train)
print("svc:")  
print("accuracy on the training subset:{:.3f}".format(svm.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(svm.score(x_test,y_test)))


'''
svc:
accuracy on the training subset:0.900
accuracy on the test subset:0.726
'''

#标准化数据
X_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)
svm1=SVC()
svm1.fit(X_train_scaled,y_train)
#改变C参数，调优,kernel表示核函数，用于平面转换，probability表示是否需要计算概率
svm1=SVC()
svm1.fit(X_train_scaled,y_train)
print("accuracy on the scaled training subset:{:.3f}".format(svm1.score(X_train_scaled,y_train)))
print("accuracy on the scaled test subset:{:.3f}".format(svm1.score(x_test_scaled,y_test)))

'''
accuracy on the scaled training subset:0.866
accuracy on the scaled test subset:0.881
'''
#改变C参数，调优,kernel表示核函数，用于平面转换，probability表示是否需要计算概率
svm2=SVC(C=10,gamma="auto",kernel='rbf',probability=True)
svm2.fit(X_train_scaled,y_train)
print("after c parameter=10,accuracy on the scaled training subset:{:.3f}".format(svm2.score(X_train_scaled,y_train)))
print("after c parameter=10,accuracy on the scaled test subset:{:.3f}".format(svm2.score(x_test_scaled,y_test)))

'''
after c parameter=10,accuracy on the scaled training subset:0.878
after c parameter=10,accuracy on the scaled test subset:0.890
'''