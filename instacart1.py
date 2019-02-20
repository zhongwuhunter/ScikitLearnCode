from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd


import numpy as np



# 获取数据
# 合并表
# 找到 user_id 与 aisle 关系
# PCA降维



def start():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    order_products = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")

    tab1 = pd.merge(aisles, products, on =["aisle_id","aisle_id"]);

    tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"]);

    tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"]);

    table = pd.crosstab(tab3["user_id"], tab3["aisle"])

    # 只取10000，性能问题。
    # data = table[:10000]
    data = table
    print(data)


#     降维
    tranfer = PCA(n_components=0.95)
    data_new = tranfer.fit_transform(data)
    print("降维后----")
    print(data_new)
start()
