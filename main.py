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

    # make more jobs fraudulent roughly about 25%
    df['fraudulent'] = [random.getrandbits(2) for i in df.index ]

    # create new column is_in_usa based off address column
    df['is_in_usa'] = df['location'].str.split(',').str[0] == 'US'



    # create ML model
    X = df[['telecommuting','has_company_logo','has_questions','employment_type', 'is_in_usa']]
    y = df['fraudulent']

    model = linear_model.LogisticRegression()

    # train and evaluate test set 
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.8)
    model.fit(X_train,y_train)
    # y_pred = model.predict(X_test)
    # print("this job is " + str(round(metrics.accuracy_score(y_test, y_pred),2)*100) + "% likely to be fake.")

    input = [ih.remote, ih.has_logo, ih.has_logo, ih.employment_type,ih.is_in_usa] 
    input = input.reshape(1,-1)
    y_pred = model.predict(input)
    print("this job is " + str(round(metrics.accuracy_score(y, y_pred),2)*100) + "% likely to be fake.")

    # visualize data
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













