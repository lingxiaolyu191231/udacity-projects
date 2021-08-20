"""
Author: Lingxiao Lyu
Date created: August 20, 2021

This module is used to test churn_library.py and log testing information
"""
import churn_library as cls
import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = import_data("./data/bank_data.csv")
        perform_eda(df)

        assert os.path.isfile("./images/eda/hist_churn_customers.png")
        assert os.path.isfile("./images/eda/hist_customer_age.png")
        assert os.path.isfile(
            "./images/eda/barplot_customer_marital_status.png")
        assert os.path.isfile("./images/eda/heatmap_features_corr.png")

        logging.info('Test perform_eda: SUCCEED')

    except AssertionError as err:
        logging.error('Test perform_eda: FAILED')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = ['Gender',
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category'
                       ]
        assert df[cat_columns].shape[0] > 0

        df = encoder_helper(df, cat_columns, 'Churn')
        logging.info('Test encoder_helper: SUCCEED')

        assert df.shape[1] == 27

        return df

    except AssertionError as err:
        logging.error(
            'Test encoder_helper: FAILED, number of features generated was not quite correct')
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = ['Gender',
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category'
                       ]
        df = encoder_helper(df, cat_columns, 'Churn')
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')
        logging.info("Test perform_feature_engineering: SUCCEED")

        assert X_train.shape[1] == 19
        assert X_test.shape[1] == 19

        return X_train, X_test, y_train, y_test

    except AssertionError as err:
        logging.error(
            "Test perform_feature_engineering: FAILED, number of features in training and tests sets not correct")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:

        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        cat_columns = ['Gender',
                       'Education_Level',
                       'Marital_Status',
                       'Income_Category',
                       'Card_Category'
                       ]
        df = encoder_helper(df, cat_columns, 'Churn')
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, 'Churn')

        train_models(X_train, X_test, y_train, y_test)
        logging.info("Test train_models: SUCCEED")
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
    except AssertionError as err:
        logging.error("Test train_models: FAILED, model files not stored ")
        raise err


if __name__ == "__main__":
    import_data = cls.import_data
    perform_eda = cls.perform_eda
    encoder_helper = cls.encoder_helper
    perform_feature_engineering = cls.perform_feature_engineering
    train_models = cls.train_models

    test_import(import_data)
    # test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
