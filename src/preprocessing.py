import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_data = pd.read_csv('data/raw/train.csv', delimiter=',')
test_data = pd.read_csv('data/raw/test.csv', delimiter=',')

# Encoding
# Process of convert categorical data to numerical data so the model can interact with it
le = LabelEncoder()
data = {'Sex': train_data.Sex.unique(), 'Cabin': train_data.Cabin.unique(), 'Embarked': train_data.Embarked.unique()}
train_data[['Sex', 'Cabin', 'Embarked']] = train_data[['Sex', 'Cabin', 'Embarked']].apply(le.fit_transform)
test_data[['Sex', 'Cabin', 'Embarked']] = test_data[['Sex', 'Cabin', 'Embarked']].apply(le.fit_transform)

# At first, it look like 'Name' and 'Ticket' won't help the classification.
train_data = train_data.drop(columns=['Name', 'Ticket'])
test_data = test_data.drop(columns=['Name', 'Ticket'])

# Create X and Y subsets
X_train = train_data.drop(columns=['Survived'])
y_train = train_data['Survived']
X_test = test_data
print(X_train.head())
print(y_train.head())
print(X_test.head())

# Save processed data
X_train.to_csv('data/processed/X_train.csv')
y_train.to_csv('data/processed/y_train.csv')
X_test.to_csv('data/processed/X_test.csv')