import itertools
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv', delimiter=',')
X_test = pd.read_csv('data/processed/X_test.csv', delimiter=',')
y_train = pd.read_csv('data/processed/y_train.csv', delimiter=',')
y_test = pd.read_csv('data/processed/y_test.csv', delimiter=',')

# RandomizedSearchCV
# Random Forest
rf_clf = RandomForestClassifier()

rf_grid = {'n_estimators': np.arange(100, 1000, step=100),
           'criterion': ['gini', 'entropy'],
           'max_depth': np.arange(10, 50, step=10),
           'max_features': ['sqrt', 'log2', None],
           'bootstrap': [True, False]}
rf_random = RandomizedSearchCV(estimator=rf_clf, param_distributions=rf_grid, cv=8, n_iter=50, verbose=3, refit=True, n_jobs=-1)
rf_random.fit(X_train, y_train.values.ravel())
estimator = rf_random.best_estimator_
print(rf_random.best_params_)
print(estimator.score(X_test, y_test))
pickle.dump(estimator, open('models/RandomForest-clf.joblib', 'wb'))

# k-NN (k-Nearest-Neighbors)
knn_clf = KNeighborsClassifier()

knn_grid = {'n_neighbors': np.arange(2, 15),
           'weights': ['uniform', 'distance'],
           'algorithm': ['ball_tree', 'kd_tree', 'brute'],
           'p': [1, 2]}
knn_random = RandomizedSearchCV(estimator=knn_clf, param_distributions=knn_grid, cv=8, n_iter=5000, verbose=3, refit=True, n_jobs=-1)
knn_random.fit(X_train, y_train.values.ravel())
estimator = knn_random.best_estimator_
print(knn_random.best_params_)
print(estimator.score(X_test, y_test))
pickle.dump(estimator, open('models/kNN-clf.joblib', 'wb'))

# SVC (Support Vector Classification)
svc_clf = SVC()

svc_grid = {'C': np.arange(2, 10, 2),
           'kernel': ['linear', 'poly'],
           'degree': np.arange(3, 5),
           'decision_function_shape': ['ovo', 'ovr']}
svc_random = RandomizedSearchCV(estimator=svc_clf, param_distributions=svc_grid, cv=8, n_iter=20, verbose=3, n_jobs=-1, refit=True)
svc_random.fit(X_train, y_train.values.ravel())
estimator = svc_random.best_estimator_
print(estimator)
print(estimator.score(X_test, y_test))
pickle.dump(estimator, open('models/SVC-clf.joblib', 'wb'))

# MLP (Multi-Layer Perceptron)
mlp_clf = MLPClassifier()

mlp_grid = {'hidden_layer_sizes': [x for x in itertools.product((10,20,30,40,50,100),repeat=3)],
           'activation': ['identity', 'logistic', 'tanh', 'relu'],
           'solver': ['lbfgs', 'sgd', 'adam'],
           'learning_rate': ['constant', 'invscaling', 'adaptive']}
mlp_random = RandomizedSearchCV(estimator=mlp_clf, param_distributions=mlp_grid, cv=8, n_iter=50, verbose=3, refit=True, n_jobs=-1)
mlp_random.fit(X_train, y_train.values.ravel())
estimator = mlp_random.best_estimator_
print(estimator)
print(estimator.score(X_test, y_test))
pickle.dump(estimator, open('models/MLP-clf.joblib', 'wb'))

# Decision Tree
dt_clf = DecisionTreeClassifier()

dt_grid = {'criterion': ['gini', 'entropy'],
           'splitter': ['best', 'random'],
           'max_depth': np.arange(10, 50, step=10),
           'max_features': ['auto', 'sqrt', 'log2', None]}
dt_random = RandomizedSearchCV(estimator=dt_clf, param_distributions=dt_grid, cv=8, n_iter=5000, verbose=3, refit=True, n_jobs=-1)
dt_random.fit(X_train, y_train.values.ravel())
estimator = dt_random.best_estimator_
print(estimator)
print(estimator.score(X_test, y_test))
pickle.dump(estimator, open('models/DT-clf.joblib', 'wb'))