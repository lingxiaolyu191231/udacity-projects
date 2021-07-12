import sys
import pandas as pd
import numpy as np
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """
    input: 
        message_filepath: csv file that stores messages
        categories_filepath: csv file that stores categories for messages

    output:
        df: a dataframe with message inforamtion and categories merged
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
    input: 
        df: a dataframe with message inforamtion and categories merged

    output:
        df: a clean dataframe with missing value replaced, categories names cleaned, 
            duplicates removed
    """

    categories = df['categories'].str.split(';',expand=True)
    row = df['categories'].str.split(';',expand=True).iloc[0,:]
    category_colnames = row.str.findall(r'[a-z]+_?[a-z]+_?[a-z]+').apply(lambda x:str(x[0])).tolist()
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast = 'integer')
    categories.loc[categories['related']>1,'related'] = 1
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset=['message'],inplace=True)
    return df

def save_data(df, database_filename):
    """
    input:
        df: a clean dataframe ready to be saved
        database_filename: file name used to save data

    output: (no output)
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql(database_filename, con=conn, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()