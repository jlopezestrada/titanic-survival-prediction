import pickle
import pandas as pd

# Load processed data
test = pd.read_csv('data/processed/test.csv', delimiter=',')

# Model prediction
rf_clf = pickle.load(open('models/RandomForest-clf.joblib', 'rb'))
y_pred = rf_clf.predict(test)

# Classification DataFrame
predictions = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})
print(predictions.shape)

# Save CSV predictions file
predictions.to_csv('predictions/submission.csv', index=False)