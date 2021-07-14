import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    """
    input: 
    database_filepath: in sql_table format

    output:
    X: input feature data
    Y: target output data
    category_names: output Y category names
    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df.iloc[:,1].astype('str')
    Y = df.iloc[:,4:]
    category_names = Y.columns.astype('str')
    
    return X,Y,category_names


def tokenize(text):
    """
    input:
        text: message in text format
    output:
        clean_tokens: clean text message
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build model pipeline
    input: None
    output: a model to be implemented with hyperparameters set up
    """
    class StartingVerbExtractor(BaseEstimator, TransformerMixin):

        def starting_verb(self, text):
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
            return 0

        def fit(self, x, y=None):
            return self

        def transform(self, X):
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged)

    pipeline = Pipeline(
        [
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('starting_verb', StartingVerbExtractor())
            ])),

            ('mclf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'mclf__estimator__warm_start':[True, False],
        'mclf__estimator__max_features':['auto','log2'],
        'mclf__estimator__criterion':['gini', 'entropy'],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the performance of the model 
    """
    y_pred = model.predict(X_test)

    ind = 0
    for cat in category_names:
        print(f'{cat} category summary:')
        print(classification_report(Y_test[cat].values, y_pred[:, ind]))
        print('------------------------------------------------------')

        ind += 1


def save_model(model, model_filepath):
    """
    use pickle to save model
    
    input:
        model: training model
        model_filepath: file path to save model

    output: None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()