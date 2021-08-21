# library doc string
"""
Udacity Machine Learning DevOps Testing and Logging Project
author: Lingxiao Lyu
date: August 20, 2021
This module consists of the core modules that are required for
- Data Import
- Data preparation
- Visualization
- Feature Engineering
- Model training
- Model save and export
- Evaluation metrics and plot
"""

# import libraries
import os
import itertools
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_churn = pd.read_csv(pth, index_col=0)
    return df_churn


def perform_eda(df_churn):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # create a new feature Churn: assign 0 for existing customer and else 1
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # plot churn vs non-churn customers and save in images folder
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    df_churn['Churn'].hist()
    plt.close(fig)
    plt.savefig('./images/eda/hist_churn_customers.png')

    # plot distribution of customer age and save in images folder
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    df_churn['Customer_Age'].hist()
    plt.close(fig)
    plt.savefig('./images/eda/hist_customer_age.png')

    # create barplot for customers at different marital status and save in
    # images folder
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    df_churn.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.close(fig)
    plt.savefig('./images/eda/barplot_customer_marital_status.png')

    # plot distribution of total transaction count
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    sns.distplot(df_churn['Total_Trans_Ct'])
    plt.close(fig)
    plt.savefig('./images/eda/displot_total_trans_ct.png')

    # plot scatterplots of age and total transcation acmount
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    sns.regplot(data=df_churn, x='Customer_Age', y='Total_Trans_Amt')
    plt.close(fig)
    plt.savefig('./images/eda/scatterplot_CustomerAge_vs_TotalTransAmount.png')

    # plot heatmap of feature correlation
    fig = plt.figure()
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_churn.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.close(fig)
    plt.savefig('./images/eda/heatmap_features_corr.png')


def encoder_helper(df_churn, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for category in category_lst:

        feature_lst = []
        feature_groups = df_churn.groupby(category).mean()[response]
        for val in df_churn[category]:
            feature_lst.append(feature_groups.loc[val])

        new_feature_name = category + '_' + response
        df_churn[new_feature_name] = feature_lst

    return df_churn


def perform_feature_engineering(df_churn, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    new_cat_columns = [cat_col + '_' + response for cat_col in cat_columns]
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]
    # define feature columns to keep as input features
    keep_cols = new_cat_columns + quant_columns

    # define X - input features, y - target feature
    X_input_data = pd.DataFrame()
    X_input_data[keep_cols] = df_churn[keep_cols]
    y_output = df_churn[response]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_input_data, y_output, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# 1st function for plotting classification report


def show_values(pc_path, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857
    By HYRY
    '''
    pc_path.update_scalarmappable()
    ax_pc = pc_path.get_axes()
    
    for p_path, color, value in itertools.zip(
            pc_path.get_paths(), pc_path.get_facecolors(), pc_path.get_array()):
        x, y = p_path.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax_pc.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

# 2nd function for plotting classification report


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    return tuple(i / inch for i in tupl[0])

# 3rd function for plotting classification report

def heatmap(
        AUC,
        title,
        xlabel,
        ylabel,
        xticklabels,
        yticklabels,
        figure_width=40,
        figure_height=20,
        correct_orientation=False,
        cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax_fig = plt.subplots()
    c = ax_fig.pcolor(
        AUC,
        edgecolors='k',
        linestyle='dashed',
        linewidths=0.2,
        cmap=cmap)

    # put the major ticks at the middle of each cell
    ax_fig.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax_fig.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax_fig.set_xticklabels(xticklabels, minor=False)
    ax_fig.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Remove last blank column
    plt.xlim((0, AUC.shape[1]))

    # Turn off all the ticks
    ax_fig = plt.gca()
    for t in ax_fig.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax_fig.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax_fig.invert_yaxis()
        ax_fig.xaxis.tick_top()

    # resize
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))

# 4th function for plotting classification report
def plot_classification_report(
        model_classification_report,
        title='Classification report',
        cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857
    '''
    lines = model_classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2: (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(
        np.array(plotMat),
        title,
        xlabel,
        ylabel,
        xticklabels,
        yticklabels,
        figure_width,
        figure_height,
        correct_orientation,
        cmap=cmap)

    plt.savefig("images/results/" + title + ".png")

# 5th function for plotting classification report


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # plot and save classification report metrics for random forest model
    plot_classification_report(
        classification_report(
            y_test,
            y_test_preds_rf),
        title='Random forest classification report for test data',
        cmap='RdBu')
    plot_classification_report(
        classification_report(
            y_train,
            y_train_preds_rf),
        title='Random forest classification report for train data',
        cmap='RdBu')

    # plot and save classification report metrics for logistic model
    plot_classification_report(
        classification_report(
            y_test,
            y_test_preds_lr),
        title='Logistic regression classification report for test data',
        cmap='RdBu')
    plot_classification_report(
        classification_report(
            classification_report(
                y_train,
                y_train_preds_lr),
            title='Logistic regression classification report for train data',
            cmap='RdBu'))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # export plot to pth
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=2000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # fit model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    lr_model = lrc

    # save best model
    joblib.dump(rfc_model, './models/rfc_model.pkl')
    joblib.dump(lr_model, './models/logistic_model.pkl')

    # plot roc curve for logistic regression
    plt.figure()
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.savefig('./images/results/logreg_roc.png')

    plt.figure(figsize=(15, 8))
    ax_gca = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax_gca, alpha=0.8)
    lrc_plot.plot(ax=ax_gca, alpha=0.8)
    plt.savefig('./images/results/randomforest_roc.png')

    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")


def main():
    """
    This is the main function starting from importing data, eda, train-test modeling and metric evaluation
    """
    # import data
    df_churn = import_data(r"./data/bank_data.csv")
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    response = 'Churn'

    # perform EDA and save plots to images folder
    perform_eda(df_churn)

    # define category features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    
    # encode categorical features
    df_churn = encoder_helper(df_churn, cat_columns, response)

    # train, fit, and save model
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_churn, response)
    train_models(X_train, X_test, y_train, y_test)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)
    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    feature_importance_plot(
        rfc_model,
        X_train,
        './images/results/rfc_features_importance.png')



if __name__ == '__main__':
    main()
