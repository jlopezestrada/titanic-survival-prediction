import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train.values.ravel())

# Save the model
pickle.dump(clf, open('model/trained_model_rf.joblib', 'wb'))