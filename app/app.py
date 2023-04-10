from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

rf_clf = pickle.load(open('models/RandomForest-clf.joblib', 'rb'))

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = {feature: int(request.form.get(feature)) for feature in request.form}
    example_data = [329,3,0,31.0,1,1,20.525,147,2]
    captured_data = list(data.values())
    captured_data.insert(0, 1)
    prediction_status = rf_clf.predict([captured_data])
    prediction = 'Passenger died'
    if prediction_status != [0]:
        prediction = 'Passenger survived.' 
    return render_template('index.html', prediction=prediction) 
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)