# Titanic Survival Prediction

This repository contains a machine learning model that predicts the likelihood of survival for passengers on the Titanic. The model is trained on a dataset that includes various features such as age, sex, and ticket class. The goal of this project is to explore the factors that contributed to survival on the Titanic and to build a model that can accurately predict survival outcomes.

## Dataset

The dataset used for this project is the famous Titanic dataset, which includes information about passengers on the Titanic such as age, sex, ticket class, and whether or not they survived. The dataset can be found in the `data` folder of this repository.

## Dependencies

To run the code in this repository, you'll need to have Python 3 installed as well as the following libraries:

- Flask==2.2.3
- matplotlib==3.5.2
- numpy==1.24.2
- pandas==2.0.0
- scikit_learn==1.2.2

You can install these libraries using `requirements.txt` by running the following command:

```pip install -r requirements.txt```

## Usage
### Command Line 

To use the trained model to make predictions on new data, you can run the `eval.py` script included in src.

### Web Application

To use the trained model to make predictions on new data using Flask web application, you can run the `app.py` script included in app.

## Conclusion

The Titanic survival prediction model built in this project achieved an accuracy of 68%. The model can be used to explore the factors that contributed to survival on the Titanic and to predict the likelihood of survival for new passengers based on their features.

