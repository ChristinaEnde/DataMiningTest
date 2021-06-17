import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#get data
import tarfile
import urllib.request

# DOWNLOAD_ROOT = "https://github.com/tleitch/handson-ml2/tree/master/"
# HOUSING_PATH = os.path.join("datasets", "housing")
# print(HOUSING_PATH)
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# print(os.path.isdir(HOUSING_PATH))

# def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
#     if not os.path.isdir(housing_path):
#         os.makedirs(housing_path)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     print(tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     print("1")
#     housing_tgz.extractall(path=housing_path)
#     print("11")
#     housing_tgz.close()
#
# fetch_housing_data()
# print("startopen")
# housing_tgz = tarfile.open("D:\code\datasets\housing\housing.tgz")
# print(type(housing_tgz))
# housing_tgz.extractall(path=HOUSING_PATH)
# housing_tgz.close()
# print('close')

import pandas as pd

# def load_housing_data(housing_path=HOUSING_PATH):
#     if not os.path.isdir(housing_path):
#      os.makedirs(housing_path)
#     csv_path = os.path.join(housing_path, "housing.csv")
#     print(csv_path)
#     return pd.read_csv(csv_path)

housing = pd.read_csv("D:\code\datasets\housing\CaliforniaHousing\cal_housing.data",header=None)
# print(housing.head(5))
column = []
with open("D:\code\datasets\housing\CaliforniaHousing\cal_housing.domain") as f:
    for i in list(f):
        column.append(i.replace('\n','').strip())
housing.columns=column
# print(housing.head(5))

#descriptive statistics
print(housing.info())
# print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# matplotlib inline
import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()
#
# # to make this notebook's output identical at every run
# np.random.seed(42)
# import numpy as np
#
# # For illustration only. Sklearn has train_test_split()
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
#
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set))
# print(len(test_set))
#
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#histogram
# housing["medianIncome"].hist()
# plt.show()

#分区
housing["income_cat"] = pd.cut(housing["medianIncome"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
print(housing["income_cat"].value_counts())

housing["income_cat"].hist()
plt.show()