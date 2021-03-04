import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


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
dataset = dataset[[ "price", "room_type", "neighbourhood_group", "neighbourhood", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]]

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

#experiment 1
#create k-1 dummy to prevent autocorrelation
dataset_1 = pd.get_dummies(dataset, columns=["neighbourhood_group","room_type"], prefix = ["nbhdg","rt"],drop_first=True)
dataset_1.drop(["neighbourhood"], axis =1, inplace= True)
data = dataset_1.loc[:, dataset_1.columns != "price"]
target = dataset_1["price"]

# regular regression
# x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
# machine = linear_model.LinearRegression().fit(x_train1, y_train1)
# print(machine.score(x_train1, y_train1))
# print(machine.coef_)
# print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))

#use ridge
# x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
# machine = linear_model.Ridge(alpha = 0.01, normalize = True).fit(x_train1, y_train1)
# print(machine.score(x_train1, y_train1))
# print(machine.coef_)
# print(list(data.columns))
# print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))
#use lasso
x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
machine = linear_model.Lasso(alpha = 0.01, normalize = True).fit(x_train1, y_train1)
print(machine.score(x_train1, y_train1))
print(machine.coef_)
print(list(data.columns))
print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))

#Experiment 2

# dataset_1 = pd.get_dummies(dataset, columns=["room_type", "neighbourhood"], prefix = ["rt","nbhd"],drop_first=True)
# dataset_1.drop(["neighbourhood_group"], axis =1, inplace= True)
# data = dataset_1.loc[:, dataset_1.columns != "price"]
# target = dataset_1["price"]
# print(list(data.columns))
#
# # regular regression
# x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
# machine = linear_model.LinearRegression().fit(x_train1, y_train1)
# print(machine.score(x_train1, y_train1))
# print(machine.coef_)
# print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))
#use ridge
# x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
# machine = linear_model.Ridge(alpha = 0.01, normalize = True).fit(x_train1, y_train1)
# print(machine.score(x_train1, y_train1))
# print(machine.coef_)
# print(list(data.columns))
# print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))
#use lasso
# x_train1, x_test1, y_train1, y_test1 = train_test_split(data, target, test_size=0.20, random_state=42)
# machine = linear_model.Lasso(alpha = 0.01, normalize = True).fit(x_train1, y_train1)
# print(machine.score(x_train1, y_train1))
# print(machine.coef_)
# print(1-(1-machine.score(x_train1,y_train1))*(len(y_train1)-1)/(len(y_train1) - x_train1.shape[1]-1))