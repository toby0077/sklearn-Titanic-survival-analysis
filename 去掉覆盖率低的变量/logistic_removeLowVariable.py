# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:39:35 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:30:24 2018

@author: Administrator
随机森林不需要预处理数据
"""
from sklearn.linear_model import LogisticRegression
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


#n_estimators表示树的个数，测试中100颗树足够
logistic=LogisticRegression()
logistic.fit(x_train,y_train)

print("logistic:")  
print("accuracy on the training subset:{:.3f}".format(logistic.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(logistic.score(x_test,y_test)))


'''
logistic:
accuracy on the training subset:0.848
accuracy on the test subset:0.875
'''