from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = {feature: request.form.get(feature) for feature in request.form}
    return data

if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=True)