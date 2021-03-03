import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

pd.set_option('display.max_columns', 500)
dataset = pd.read_csv("hw1.csv")
# print(dataset.dtypes)
# print(dataset.head(7))
# print(dataset[["price"]].isnull().sum())
# print(dataset[["room_type"]].isnull().sum())
# print(dataset[["neighbourhood_group"]].isnull().sum())
# print(dataset[["neighbourhood"]].isnull().sum())
# print(dataset[["minimum_nights"]].isnull().sum())
# print(dataset[["number_of_reviews"]].isnull().sum())
# print(dataset[["last_review"]].isnull().sum())
# print(dataset[["reviews_per_month"]].isnull().sum())
# print(dataset[["availability_365"]].isnull().sum())
# print(dataset[["calculated_host_listings_count"]].isnull().sum())

#replacing NA with just 0s in reviews per month
dataset.fillna({"reviews_per_month":0}, inplace=True)

#subsetting the dataset with 10 variables and ID
dataset = dataset[["id", "price", "room_type", "neighbourhood_group", "neighbourhood", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]]

print(dataset.isnull().sum())

print(dataset["neighbourhood_group"].value_counts())

#Here I tested price and minimum_nights variables to find outliers
dataset["price"].plot.hist(bins=10)
print(dataset["price"].describe())
#x = dataset["price"].hist()
#plt.show()
#deciding cutoff point
print(dataset[dataset["price"]>300])
#subsetting and cutoff prices over 275.
dataset = dataset[dataset["price"]<300]

dataset["minimum_nights"].plot.hist(bins=10)
print(dataset["minimum_nights"].describe())
dataset = dataset[dataset["price"]<14]
region_dummies = pd.get_dummies(dataset['region'])
occ_dummies = pd.get_dummies(dataset['occ'])
