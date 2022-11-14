# import nltk 
# from nltk.tokenize import RegexpTokenizer 
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection
import random
# from wordcloud import WordCloud
from InputHelper import InputHelper

def main():
    print('starting program')
    # get some input
    ih = InputHelper()

    # parse csv 
    df = pd.read_csv('fake_job_postings.csv')
    # wrangle data
    df=tokenize_job_title_col(df)

    # make more jobs fraudulent this should randomly make half of the jobs fraudulent
    df['fraudulent'] = [random.getrandbits(1) for i in df.index]

    # create new column is_in_usa based off address column
    df['is_in_usa'] = df['location'].str.split(',').str[0] == 'US'

    # df['fraudulent'] = 1
    # df['fraudulent'][0] = 0

    # check data
    # print(df.columns)
    # print(df['employment_type'].unique().tolist())
    # print(df['is_in_usa'])

    # visualize data


    # create ML model
    X = df[['telecommuting','has_company_logo','has_questions','employment_type']]
    y = df['fraudulent']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3)

    model = linear_model.LogisticRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    # y_pred = model.predict([[ih.remote, ih.has_logo, ih.has_logo, ih.employment_type]])
    # y_pred = model.predict([[1, 1, 1, 1]])
    # print(y_pred)
    # print(X.shape[0])
    # print(y.shape[0])
    # print(df.loc[df['fraudulent'] == 1]['description'])

    # print(df[df['fraudulent']==1].shape[0])

def is_in_usa(row):
    if row['location'].split(',')[0] == 'US':
        val = 1
    else: val = 0

def tokenize_job_title_col(dataframe):
    dataframe['employment_type'].fillna(5, inplace=True)
    dataframe.replace({'employment_type': {
        'Full-time': 1,
        'Part-time': 2,
        'Contract': 3,
        'Temporary': 4,
        'Other': 5,
    }},inplace=True)
    dataframe['employment_type'].astype(int)
    return dataframe

if __name__ == '__main__':
    main()













