#学生学习时间与成绩的关系
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#数据预处理没什么说的
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#将训练集用简单线性回归来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#预测结果
Y_pred = regressor.predict(X_test)

#训练集可视化
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

#测试集可视化
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

"""
从散点图来看，大致是正相关的，但是有个别异常点，
即有某些学习时间很短但是成绩仍然很好的人
"""