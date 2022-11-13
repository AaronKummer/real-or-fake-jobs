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
    ih = InputHelper()

    # parse csv
    df = pd.read_csv('fake_job_postings.csv')
    

    # check data
    # print(df.columns)
    # print(df['employment_type'].unique().tolist())







    print('done')

if __name__ == '__main__':
    main()













