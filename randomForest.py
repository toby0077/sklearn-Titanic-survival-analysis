# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:30:24 2018

@author: Administrator
随机森林不需要预处理数据
"""
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

trees=1000
max_depth=10
#n_estimators表示树的个数，测试中100颗树足够
forest=RandomForestClassifier(n_estimators=trees,random_state=0,max_depth=max_depth)
forest.fit(x_train,y_train)

print("random forest with %d trees:"%trees)  
print("accuracy on the training subset:{:.3f}".format(forest.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(forest.score(x_test,y_test)))
#print('Feature importances:{}'.format(forest.feature_importances_))

names=features.columns
importance=forest.feature_importances_
zipped = zip(importance,names)
list1=list(zipped)

list1.sort(reverse=True)
#print(list1)



n_features=data_dummies.shape[1]
plt.barh(range(n_features),forest.feature_importances_,align='center')
plt.yticks(np.arange(n_features),features)
plt.title("random forest with %d trees,%dmax_depth:"%(trees,max_depth))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


'''
random forest with 1000 trees:
accuracy on the training subset:0.983
accuracy on the test subset:0.878


random forest with 1000 trees,max_depth=4:
accuracy on the training subset:0.854
accuracy on the test subset:0.884

random forest with 1000 trees,max_depth=5:
accuracy on the training subset:0.853
accuracy on the test subset:0.887

random forest with 1000 trees,max_depth=9
accuracy on the training subset:0.871
accuracy on the test subset:0.890
'''