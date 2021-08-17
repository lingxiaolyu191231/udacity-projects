# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project employs a dataset of bank customers with their demographic information and their activity history with the bank account. The goal of this project is to build two classification models - random forest and logistic regression - to predict whether a customer is likely to churn or not. Unit testing and logging is a essential part of building machine learning models. Thus, a separate test python file is created and logging function is embedded for debugging purpose. The model is built in churn_library.py file, which includes all necessary functions. THe __Data__ folder contains the raw data that would be used to train and test models. The __images__ folder contains plots of exploratory analysis, classification model output, and feature importance plots. In the __model__ folder, there is two files that save model metadata. 

## Running Files
- Step 1: pip install all python packages, including shap, joblib, pandas, numpy,matplotlib,seaborn, sklearn

- Step 2: run _ipython churn_script_logging_and_tests.py_ 

- Step 3: open __logs__ folder and open __churn_library.log__ file to check whether there is error shown

- Step 4: run _ipython churn_library.py_ to build models and obtain the model outputs

- Step 5 (optional): open __images__ folder and open two subfolders, which are __eda__ and __results__. __eda__ folder includes all plots from exploratory analysis. __results__ folder contains plots such as classification reports and feature importance plots.


