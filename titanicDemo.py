
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

titanic = pd.read_csv("./titanic.csv")

# 筛选特征
x = titanic[["pclass", "age", "sex"]]
y = titanic["survived"]

# 数据处理

# 缺失值处理
x.fillna(x["age"].mean(), inplace=True)

# 转换成字典
x = x.to_dict(orient="records")


# 数据集划分划分
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)



transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 3）决策树预估器
estimator = DecisionTreeClassifier(criterion="entropy", max_depth=8)
estimator.fit(x_train, y_train)

# 4）模型评估
# 方法1：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 可视化决策树  http://www.webgraphviz.com/
export_graphviz(estimator, out_file="titanic.dot", feature_names=transfer.get_feature_names())



print("************************随机森林**************************************");

# 随机森林
estimator = RandomForestClassifier()
# 加入网格搜索与交叉验证
param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200],
                   "max_depth": [5, 8 ,15, 25, 30]}

estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
estimator.fit(x_train, y_train)

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

