import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder = LabelEncoder()
X[:, 3] = LabelEncoder.fit_transform(X[:, 3])
onehotencoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
# 新版本可以自动舍弃独热编码的第一行内容，也就是使用drop='first'参数来自动舍弃
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = X[: , 1:]
# 原本写法比较臃肿，现版本更加简洁方便，由于原仓库比较老，所以好多东西都有了新版本可以使用，
# 在后面我都会把原来的和现在不同的进行修改和比较。
X = onehotencoder.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(y_train, regressor.predict(X_train), color='blue', label='Predicted (Train)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', label='Ideal Fit')
plt.title('Training Data: Predicted vs Actual')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.show()

# 绘制测试数据的真实值 vs 预测值
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='green', label='Predicted (Test)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit')
plt.title('Testing Data: Predicted vs Actual')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.legend()
plt.show()

"""
其实由结果可以明显看出这个模型的拟合效果比较好，事实上我也渐渐发现数据处理才是整个机器学习中最重要的部分
"""



