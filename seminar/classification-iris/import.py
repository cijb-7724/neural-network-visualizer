from sklearn import datasets
import csv
iris = datasets.load_iris()

print(iris.target_names)
print(iris.feature_names)
X = iris.data
y = iris.target
print(X)
print(y)

with open('data/iris_datasets_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(X)
with open('data/iris_datasets_target.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(y)
    