# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:04:10 2018

@author: Administrator
"""
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

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
X_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
#变量名
names=features.columns

#调参
list_average_accuracy=[]
depth=range(1,30)
for i in depth:
    #max_depth=4限制决策树深度可以降低算法复杂度，获取更精确值
    tree= DecisionTreeClassifier(max_depth=i,random_state=0)
    tree.fit(X_train,y_train)
    accuracy_training=tree.score(X_train,y_train)
    accuracy_test=tree.score(x_test,y_test)
    average_accuracy=(accuracy_training+accuracy_test)/2.0
    #print("average_accuracy:",average_accuracy)
    list_average_accuracy.append(average_accuracy)
    
max_value=max(list_average_accuracy)
#索引是0开头，结果要加1
best_depth=list_average_accuracy.index(max_value)+1
print("best_depth:",best_depth)

best_tree= DecisionTreeClassifier(max_depth=best_depth,random_state=0)
best_tree.fit(X_train,y_train)
accuracy_training=best_tree.score(X_train,y_train)
accuracy_test=best_tree.score(x_test,y_test)

print("decision tree:")    
print("accuracy on the training subset:{:.3f}".format(best_tree.score(X_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(best_tree.score(x_test,y_test)))

'''
best_depth: 19
decision tree:
accuracy on the training subset:0.976
accuracy on the test subset:0.860
'''

#绘图，显示因子重要性
n_features=x.shape[1]
plt.barh(range(n_features),best_tree.feature_importances_,align='center')
plt.yticks(np.arange(n_features),features)
plt.title("Decision Tree:")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

#生成一个dot文件，以后用cmd形式生成图片
export_graphviz(best_tree,out_file="Titanic.dot",class_names=['death','live'],feature_names=names,impurity=False,filled=True)
