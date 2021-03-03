import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics 
dataset = pd.read_csv("wage_dataset.csv")

print(dataset)

dataset = dataset.sample(frac=1).reset_index(drop=True) #draw 1% of the dataset

region_dummies = pd.get_dummies(dataset['region'])
occ_dummies = pd.get_dummies(dataset['occ'])
exp2 = dataset['exp'].astype(int)**2


data = pd.concat([
	dataset[['ability', 'age', 'female', 'education', 'exp']],
    region_dummies, 
    occ_dummies,
    exp2 
    ], axis = 1).values
target = dataset.iloc[:,1].values

print(data)
print(target)

#Set test and training dataset
kfold_object = KFold(n_splits=4)#how many chucked dataset

kfold_object.get_n_splits(data)

#print(kfold_object)
i = 0
for training_index, test_index in kfold_object.split(data):
	print(i)
	i = i + 1
	print("training:", training_index)
	print("test:", test_index)
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training,target_training)
	new_target = machine.predict(data_test)
	print("R2 score:", metrics.r2_score(target_test,new_target))