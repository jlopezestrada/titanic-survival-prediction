import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# Random Forest Classifier
rf_clf = RandomForestClassifier()
# rf_clf.fit(X_train, y_train.values.ravel())

# k-NN Classifier
knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_train.values.ravel())

# SVC Classifier
svc_clf = SVC()
# svc_clf.fit(X_train, y_train.values.ravel())

# MLP Classifier
mlp_clf = MLPClassifier()
# mlp_clf.fit(X_train, y_train.values.ravel())

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
# dt_clf.fit(X_train, y_train.values.ravel())

# # Save the models
# pickle.dump(rf_clf, open('models/RandomForest-clf-Titanic.joblib', 'wb'))
# pickle.dump(knn_clf, open('models/kNN-clf-Titanic.joblib', 'wb'))
# pickle.dump(svc_clf, open('models/SVC-clf-Titanic.joblib', 'wb'))
# pickle.dump(mlp_clf, open('models/MLP-clf-Titanic.joblib', 'wb'))
# pickle.dump(dt_clf, open('models/DT-clf-Titanic.joblib', 'wb'))

# Cross validation
k_folds = KFold(n_splits = 10)
# scores = {'RF': cross_val_score(rf_clf, X_train, y_train.values.ravel(), cv = k_folds),
#           'kNN': cross_val_score(knn_clf, X_train, y_train.values.ravel(), cv = k_folds),
#           'SVC': cross_val_score(svc_clf, X_train, y_train.values.ravel(), cv = k_folds),
#           'MLP': cross_val_score(mlp_clf, X_train, y_train.values.ravel(), cv = k_folds),
#           'DT': cross_val_score(dt_clf, X_train, y_train.values.ravel(), cv = k_folds)}

# RandomizedSearchCV
# Random Forest
rf_grid = {'n_estimators': np.arange(100, 1000, step=100),
           'criterion': ['gini', 'entropy'],
           'max_depth': np.arange(10, 50, step=10),
           'max_features': ['sqrt', 'log2', None],
           'bootstrap': [True, False]}
rf_random = RandomizedSearchCV(estimator=rf_clf, param_distributions=rf_grid, n_iter=25, cv=25, verbose=3, n_jobs=-1)
rf_random.fit(X_train, y_train.values.ravel())
print(rf_random.best_params_)
print(rf_random.score(X_test, y_test))

# [print(score[0], score[1].mean()) for score in scores.items()]