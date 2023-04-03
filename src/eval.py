import pickle
import pandas as pd

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# Load trained model
clf = pickle.load(open('model/trained_model_rf.joblib', 'rb'))

# Evaluation
eval = clf.score(X_test, y_test)
print(eval)