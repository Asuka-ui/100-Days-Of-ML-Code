import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#导入数据集
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #iloc参数前面是行后面是列，’:‘表示全部，-1表示倒数第一行，选取范围和python切片一致
Y = dataset.iloc[:, 3].values

#处理缺失项
imputer = SimpleImputer(missing_values = 'np.nan', strategy = 'mean') #这一步仅仅是提供填充策略，并没有真正的改变数据
imputer = imputer.fit(X[:, 1:3]) #这一步仅仅是根据我上一步决策的方法来拟合适当的填充值，也没有实际的改变我选取的数据

X[:, 1:3] = imputer.transform(X[:, 1:3])

#解析分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#这里先使用labelencoder转化成数字编码，然后再使用 onehotencoder转化为独热

onehotcoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = onehotcoder.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y) #一般特征数据（X）使用独热而目标变量（Y）通常使用labelencoder

#划分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)