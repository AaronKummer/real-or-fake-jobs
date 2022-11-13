# import nltk 
# from nltk.tokenize import RegexpTokenizer 
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from wordcloud import WordCloud
from InputHelper import InputHelper

def main():
    print('starting program')
    # get some input
    # ih = InputHelper()

    # parse csv
    df = pd.read_csv('fake_job_postings.csv')
    df=tokenize_job_title_col(df)

    # check data
    # print(df.columns)
    print(df['employment_type'].unique().tolist())



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













