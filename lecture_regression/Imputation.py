import pandas

dataset_missing = pandas.read_csv("dataset_missing.csv", dtype= 'object')

print(dataset_missing)

print(dataset_missing.isnull().sum())

for i in range(2000,2011):
    print("year:", str(i))
    print(dataset_missing [dataset_missing['year']==str(i)].isnull().sum())

def data_imputation(dataset_missing, impute_target_name, impute_data):
    impute_target = dataset[impute_target_name]
    sub_dataset = pandas.concat([impute_target, impute_data], axis = 1) #subsetting dataset to run imputation
    data = sub_dataset.loc[:, sub_dataset.columns != impute_target_name][sub_dataset[impute_target_name].notnull()].values
    target = dataset[impute_target_name][sub_dataset[impute_target_name].notnull()].values
    machine = linear_model.LinearRegression()
    machine.fit(data,target)
    return machine.predict(sub_dataset.loc[:, sub_dataset.columns != impute_target_name].values)

dataset["ability", "age", "female", "education", "exp"]

data_imputation(dataset_missing, "earning", impute_data)

