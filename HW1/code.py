import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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
dataset = dataset[dataset["minimum_nights"]<14]

dataset_1 = pd.get_dummies(dataset, columns=["neighbourhood_group","room_type"], prefix = ["nbhdg","rt"],drop_first=True)
dataset_1.drop(["neighbourhood"], axis =1, inplace= True)
data = dataset_1.loc[:, dataset_1.columns != "price"]
target = dataset_1["price"]
# machine = linear_model.LinearRegression()
# machine.fit(data,target)
# print(machine.coef_)

#use ridge
for power in range (1,7):
    print ("power:", )
# machine = linear_model.Ridge(alpha = 0.001, normalize=True)
# machine.fit(data,target)
# print(machine.coef_)

#use lasso
machine = linear_model.Lasso(alpha = 0.001, normalize=True)
machine.fit(data,target)

print(machine.coef_)


# dataset_1 = pd.get_dummies(dataset, columns=["neighbourhood_group","room_type", "neighbourhood"], prefix = ["nbhdg","rt","nbhd"],drop_first=True)


# machine = linear_model.LinearRegression()
# machine.fit(data,target)
# print(machine.coef_)

# i = 0
# for training_index, test_index in kfold_object.split(data):
# 	print(i)
# 	i = i + 1
# 	print("training:", training_index)
# 	print("test:", test_index)
# 	data_training = data[training_index]
# 	data_test = data[test_index]
# 	target_training = target[training_index]
# 	target_test = target[test_index]
# 	machine = linear_model.LinearRegression()
# 	machine.fit(data_training,target_training)
# 	new_target = machine.predict(data_test)
# 	print("R2 score:", metrics.r2_score(target_test,new_target))