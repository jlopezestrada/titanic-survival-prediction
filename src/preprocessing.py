import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Process empty or NaN values
train_data = train_data.dropna()
# test_data = test_data.apply(lambda x: x.fillna(x.mean()), axis=0)
# test_data = test_data.apply(lambda x: x.fillna(0), axis=0)
test_data = test_data.apply(lambda x: x.fillna(x.median()), axis=0)

# Create X and Y subsets
# Entire dataset used for Kaggle Scoring
X_train_full = train_data.drop(columns=['Survived'])
y_train_full = train_data['Survived']
test = test_data

# Train split to personal score purposes
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
print("X_train", X_train.shape)
print("X_test", X_test.shape)
# print(y_train.head())

# Save processed data
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)