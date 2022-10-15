from sklearn import neighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_titanic.csv')
data.head()

# remove columns that do not carry the necessary information
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

# take the expected result into a separate variable
y = data['Survived']
data = data.drop(["Survived"], axis = 1)

# fill in the blanks
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna("S")

# replacing object data type with int
data['Sex'] = data['Sex'].astype('category')
data['Sex'] = data['Sex'].cat.codes

data = pd.get_dummies(data, columns = ['Embarked'])


from sklearn.model_selection import train_test_split

train_data, val_data, train_y, val_y = train_test_split(data, y, test_size = 0.3)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

test_data = pd.read_csv('test_titanic.csv')
test_data.head(3)

# test data processing
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

test_data['Sex'] = test_data['Sex'].astype('category')
test_data['Sex'] = test_data['Sex'].cat.codes

test_data = pd.get_dummies(test_data, columns = ['Embarked'])


# model training
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(data, y)

test_predicted = knn.predict(test_data)

# saving predicted values
test_predicted = pd.DataFrame({'Survived':test_predicted})
test_predicted['PassengerId'] = list(range(892, 892 + len(test_data)))
test_predicted.to_csv('test_predicted_titanic.csv', index = False)

print(test_predicted)