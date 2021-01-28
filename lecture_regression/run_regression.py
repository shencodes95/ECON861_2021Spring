import pandas
from sklearn import linear_model

dataset = pandas.read_csv("dataset_3_outputs.csv")

print(dataset)

target = dataset.iloc[:, 1].values  # zero column is one

print(target)

data = dataset.iloc[:, 3:9].values  # second number is not inclusive

print('data')

machine = linear_model.LogisticRegression()

machine.fit(data, target)

print(machine.coef_)

print(machine)

new_data = [
    [0, 1, 2, 3, 4, 5],
    [5, 4, 2, 3, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]  # big square bracket represent a variable

new_target = machine.predict(new_data)

print(new_target)