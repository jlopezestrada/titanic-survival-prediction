import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def show_clf_report(y_test, y_pred):
    return classification_report(y_test, y_pred)

def show_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)

def model_evaluation(model):
    # TODO
    return 0

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# Load trained models
# Random Forest
rf_clf = pickle.load(open('models/RandomForest-clf.joblib', 'rb'))

# # k-NN
# knn_clf = pickle.load(open('models/kNN-clf-Titanic.joblib', 'rb'))

# # SVC
# svc_clf = pickle.load(open('models/SVC-clf-Titanic.joblib', 'rb'))

# # MLP
# mlp_clf = pickle.load(open('models/MLP-clf-Titanic.joblib', 'rb'))

# # Decision Tree
# dt_clf = pickle.load(open('models/DT-clf-Titanic.joblib', 'rb'))

# Evaluation
# rf_eval = rf_clf.score(X_test, y_test)
y_pred = rf_clf.predict(X_test)
# knn_eval = knn_clf.score(X_test, y_test)
# svc_eval = svc_clf.score(X_test, y_test)
# mlp_eval = mlp_clf.score(X_test, y_test)
# dt_eval = dt_clf.score(X_test, y_test)

# print(f'Random Forest accuracy: {rf_eval}')
# print(f'k-NN accuracy: {knn_eval}')
# print(f'SVC accuracy: {svc_eval}')
# print(f'MLP accuracy: {mlp_eval}')
# print(f'Decision Tree accuracy: {dt_eval}')

print(classification_report(y_test, y_pred))