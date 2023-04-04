import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def model_evaluation(model, X_test):
    y_pred = model.predict(X_test)
    print(f'Model: {model}')
    print('Classification Report')
    print('=====================')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix')
    print('=====================')
    print(confusion_matrix(y_test, y_pred))

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# Load trained models
# Random Forest
rf_clf = pickle.load(open('models/RandomForest-clf.joblib', 'rb'))
model_evaluation(rf_clf, X_test)

# k-NN
knn_clf = pickle.load(open('models/kNN-clf.joblib', 'rb'))
model_evaluation(knn_clf, X_test)

# SVC
svc_clf = pickle.load(open('models/SVC-clf.joblib', 'rb'))
model_evaluation(svc_clf, X_test)

# MLP
mlp_clf = pickle.load(open('models/MLP-clf.joblib', 'rb'))
model_evaluation(mlp_clf, X_test)

# # Decision Tree
dt_clf = pickle.load(open('models/DT-clf.joblib', 'rb'))
model_evaluation(dt_clf, X_test)