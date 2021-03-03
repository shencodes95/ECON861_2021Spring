import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

dataset_missing = pandas.read_csv("dataset_missing.csv", dtype= 'object')

print(dataset_missing)

print(dataset_missing.isnull().sum())

#for i in range(2000,2011):
#   print("year:", str(i))
#    print(dataset_missing [dataset_missing['year']==str(i)].isnull().sum())

def data_imputation(dataset, impute_target_name, impute_data):
    impute_target = dataset[impute_target_name]
    sub_dataset = pandas.concat([impute_target, impute_data], axis = 1) #subsetting dataset to run imputation
    data = sub_dataset.loc[:, sub_dataset.columns != impute_target_name][sub_dataset[impute_target_name].notnull()].values
    target = dataset[impute_target_name][sub_dataset[impute_target_name].notnull()].values
    kfold_object = KFold(n_splits=4)
    kfold_object.get_n_splits(data)

    i =0
    for training_index, test_index in kfold_object.split(data):
        i = i+1
        print("case:", str(i))
        data_training = data[training_index]
        data_test = data[test_index]
        target_training = target[training_index]
        target_test = target[test_index]
        one_fold_mahine = linear_model.LinearRegression()
        one_fold_mahine.fit(data_training, target_training)
        new_target = one_fold_mahine.predict(data_test)
        print(metrics.mean_absolute_error(target_test, new_target))

    machine = linear_model.LinearRegression()
    machine.fit(data,target)
    return machine.predict(sub_dataset.loc[:, sub_dataset.columns != impute_target_name].values)

#impute_data = dataset_missing[["ability", "age", "female", "education", "exp"]]  #why2brackets? does not tak a string as input, we need to form an array

region_dummies = pandas.get_dummies(dataset_missing["region"])
occ_dummies = pandas.get_dummies(dataset_missing["occ"])

impute_data = pandas.concat([dataset_missing[["ability", "age", "female", "education", "exp"]], region_dummies], axis = 1)

new_earnings = pandas.DataFrame(data_imputation(dataset_missing, "earnings", impute_data))
new_earnings.rename(columns = {0: "earnings_imputed"}, inplace = True)

dataset_missing = pandas.concat([dataset_missing, new_earnings], axis = 1)
dataset_missing['earnings_missing'] = dataset_missing['earnings'].isnull()

print(dataset_missing)
print(dataset_missing.isnull().sum())

dataset_missing['earnings'].fillna(dataset_missing['earnings_imputed'], inplace=True)

print(dataset_missing.isnull().sum())

dataset_missing.to_csv("data_not_missing.csv")