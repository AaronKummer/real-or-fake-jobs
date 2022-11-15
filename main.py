import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import metrics, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import random
from InputHelper import InputHelper

def main():
    print('starting program')
    # get some input
    ih = InputHelper()
    # parse csv 
    df = pd.read_csv('fake_job_postings.csv')
    # wrangle data
    df=tokenize_job_title_col(df)
    # make more jobs fraudulent 
    df['fraudulent'] = [random.getrandbits(3) for i in df.index ]

    # create new column is_in_usa based off address column
    df['is_in_usa'] = df['location'].str.split(',').str[0] == 'US'

    # create ML model
    X = df[['telecommuting','has_company_logo','has_questions','employment_type', 'is_in_usa']]
    y = df['fraudulent']
    input = [ih.remote, ih.has_logo, ih.has_logo, ih.employment_type,ih.is_in_usa]
    input = np.array(input)
    input = input.reshape(1,-1)

    # evaluate accuracy
    training_model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
    training_model.fit(X_train,y_train)
    v_pred = training_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, v_pred)
    print(accuracy)

    model = DecisionTreeClassifier()
    model.fit(X,y)
    y_pred = model.predict(input)
    answer = 'fake' if y_pred==1 else 'real'
    print(answer)
    print('the job is ' + str( round(accuracy*100,2)) + '% likely to be ' + answer)

    # visualize data
    # print(X.shape[0])
    # print(y.shape[0])
    # print(df.loc[df['fraudulent'] == 1]['description'])
    # print(df[df['fraudulent']==1].shape[0])
    # print(df['fraudulent'].unique())

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













