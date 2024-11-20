import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
"""
该数据集包含了社交网络中用户的信息。这些信息涉及用户ID,性别,年龄以及预估薪资。
一家汽车公司刚刚推出了他们新型的豪华SUV，我们尝试预测哪些用户会购买这种全新SUV。
并且在最后一列用来表示用户是否购买。我们将建立一种模型来预测用户是否购买这种SUV，该模型基于两个变量，分别是年龄和预计薪资。
因此我们的特征矩阵将是这两列。我们尝试寻找用户年龄与预估薪资之间的某种相关性，以及他是否购买SUV的决定。
"""
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# 这里观察数据集可以发现，这次实验需要的数据全是连续变量，也就是说无需再使用onehotencoder和labelencoder进行处理
# 因此任何的代码都是基于具体需求的，没有需求而写出的代码也是没有意义的。

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 这里对训练集拟合再变换，而对于测试集只是进行了变换，
# 这是因为测试集的目的是为了保证训练集的可靠性，因此需要和训练集使用一样的统计数据。
# 避免再次计算统计量导致训练集和测试集分布不一致，造成不公平的模型评估

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot = True, fmt = 'd', cbar = 'Blues',
            xticklabels = ['Predicted 0', 'Predicted 1'], yticklabels = ['True 0', 'True 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.show()

# 这里是一个混淆矩阵，它能直观地显示分类模型在每个类别上的预测结果，从而帮助分析模型的优劣和错误分布。
# 分类的类型有以下四种：
# 真正类“TP”
# 假正类“FP”（也就是被误判为正的，可以叫做“误报”）
# 假负类“FN”（也就是被误判为负的，可以叫做“漏报”）
# 真负类“TN”

# 由以上数据可以计算一些模型的性能指标：
# ACC(准确率) = (TP + TN) / (TP + FP + FN + TN)
# Precision(精确率) = TP / (TP + FP) 指被预测为正类的里面，实际正类的比例
# Recall(召回率) = TP / (TP + FN) 实际正类里面被正确预测为正类的比例

# 如果一个模型的FP很多，那么认为这个模型比较激进
# 如果一个模型的FN很多，认为这个模型比较保守

# 可视化
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
    np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# 这里使用的arange是为了在x1上划分网格点，X测试集中有多少个特征列就需要多少个arange来划分，最后通过meshgrid来统一。

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.25, cmap = ListedColormap(('red', 'blue')))

# 将原来产生的二维数组X1,X2先展开成一维数组，然后将两个一维数组结合出[x1, x2]的坐标，此时还是一维数组形式，reshape转回二维数组。
# 设置背景颜色不透明度为75%，颜色为red和blue。

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# 设置整个画面边界，防止超出。

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(i), label=j)
# 这个部分的代码有些不好理解,主要是函数功能嵌套的太复杂，分开理解
# 首先np.unique(y_set)返回y_set数组中唯一的标签此处提取出来为[0, 1]
# 那么这里就会历经两轮循环也就是当类别为0涂红，类别为1涂蓝

plt. title('LOGISTIC(Training set)')
plt. xlabel('Age')
plt. ylabel('Estimated Salary')
plt. legend()
plt. show()

X_set,y_set=X_test,y_test
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.25, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'blue'))(i), label=j)

plt. title('LOGISTIC(Test set)')
plt. xlabel('Age')
plt. ylabel('Estimated Salary')
plt. legend()
plt. show()