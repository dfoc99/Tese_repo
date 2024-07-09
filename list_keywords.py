
import dataframe
import KeyBERT_model
import pandas as pd
import split_dataframe

import numpy as np
import json
import re
import pandas as pd
import networkx as nx
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Downloading required NLTK resources
nltk.download("popular")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from urllib.parse import urlparse
import warnings



def main(df):
    # df -> dataframe 

    # Drop any duplicates
    df = df.drop_duplicates(subset=["Abstract"])
    print("data_articles shape: ",df.shape)

    # Reset dataframe's indices
    df.reset_index(drop=True)

    df["Keywords"] = df["Keywords"].astype(str)
    # It returns a list of keywords in a class <list>
    # It returns all strings in the keywords column, for each line
    list_of_keywords = list(df['Keywords'])
    print("Number of entries of list_of_keywords: ",len(list_of_keywords))


    # It returns a nested array where in which list it contains keywords for a given article
    new_keywords_list = [article.split(", ") for article in list_of_keywords]
    final_keywords_list = []
    for article in new_keywords_list:
        final_article = []
        for item in article:
            final_article.extend(item.split(" and "))
        final_keywords_list.append(final_article)

    # list of abstracts    
    list_abstract = list(df["Abstract"])


    return final_keywords_list, list_abstract,df