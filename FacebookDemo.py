import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# 1 获取数据
# 2 划分数据集
# 3 特征工程: 文本特征提取 -tfidf
# 4 朴素贝叶斯算法流程
# 5 模型评估

# 加载数据
data = pd.read_csv("./FacebookData/train.csv")
# 缩小范围
# data = data.query("x < 2.5 & x > 2 & y < 1.5 & y > 1.0")


# 时间处理
time_value = pd.to_datetime(data["time"], unit="s")
date = pd.DatetimeIndex(time_value)
data["day"] = date.day
data["weekday"] = date.weekday
data["hour"] = date.hour

# 过滤地点
#  groupby 分组， 统计 place_id 相同的数据有多少  https://www.jianshu.com/p/42f1d2909bb6
place_count = data.groupby("place_id").count()["row_id"]
# 过滤 place_count 小于3 的位置，也就是过滤掉签到小于3的位置
data_final = data[data["place_id"].isin(place_count[place_count > 3].index.values)]


# 筛选特征
x = data_final[["x", "y", "accuracy", "weekday", "hour"]]
y = data_final["place_id"]

# 划分数据
x_train, x_test, y_train, y_test = train_test_split(x, y)

f


# 特征工程
# x_train = x_train.astype(float)
# x_test = x_test.astype(float)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)



# 最后我去B

# KNN算法预估器

estimator = KNeighborsClassifier()
# param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}

param_dict = {"n_neighbors": [5, 7, 9, 11]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=2)

# estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
estimator.fit(x_train, y_train)

# 5）模型评估
# 方法1：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 最佳参数：best_params_
print("最佳参数：\n", estimator.best_params_)
# 最佳结果：best_score_
print("最佳结果：\n", estimator.best_score_)
# 最佳估计器：best_estimator_
print("最佳估计器:\n", estimator.best_estimator_)
# 交叉验证结果：cv_results_
print("交叉验证结果:\n", estimator.cv_results_)






























