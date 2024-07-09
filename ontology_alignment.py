# These functions are designed to treat this specific data on keywords and entities. It combines split techniques with regex methods to clean data.
# Each function's purpose is specified at the beginning of each function.

import dataframe
import KeyBERT_model
import pandas as pd
import split_dataframe

import numpy as np
import re
import pandas as pd
import networkx as nx
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import json
import ast

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


def separate_strings(input_array):
  """
   Should read this array and, verifying that contains multiple strings,
   separates them and append them into a foreign list.
  """
  foreign_list = []
  for entry in input_array:
    words = entry.split(" ")
    for word in words:
      foreign_list.append(word)
  return foreign_list

def separate_special_characters(input_array):
  """
   Should remove all strings with special characters such as "string1:".
  """
  cleaned_list = []
  special_list = []
  for item in input_array:
      if not re.match(r'^[a-zA-Z0-9 ]*$', item):
        special_list.append(item)
      else:
        cleaned_list.append(item)
  return cleaned_list,special_list



def treat_special_characters(colon_list):
    """
    This function will treat all strings that were separated
    in remove_special_characters into "special_list" array.
    Essentially, we will have two conditions.
    One for each word that contains the special character ":" at the end of the word. For example "X-ray:".
    Other for each word that contains any special characters between them. For example "X-ray".
    """

    no_colon_list = []
    for string in colon_list:
        matched_part = re.search(r'[a-zA-Z0-9-:]+', string)
        if matched_part:
            colon_string = matched_part.group()[-1]
            desired_part = matched_part.group()
            if colon_string == ":":
                desired_part = matched_part.group()[:-1]
            no_colon_list.append(desired_part)

    return no_colon_list


def cleaned_list_plus_no_colon_list(array1,array2):
  """
  Adds two arrays
  """
  final_list = array2 + array1
  return final_list


def replace_strings(arr):
  """
  This function will replace all strings "-" with "_", giving an important notation for some strings,
  such as "Klein-Gordon" or "X-rays" which typically it's words that don't make any sense separated.
  """
  # Iterate through the array
  for i in range(len(arr)):
    # If the element contains "-" or ":", replace them with "_"
    if "-" in arr[i] or ":" in arr[i]:
        arr[i] = arr[i].replace("-", "_").replace(":", "_")
  return arr

def remove_special_and_numeric_strings(arr):
  """
  This function removes strings that contain only special characters or only digits. So for example the string "04" should be removed because it only contains digits.
  The string "_" also should be removed, because it only contains special characters.
  There's another example that should be removed, which is the string "04_70", because it's a mix of both.
  If we encountered the string "D04_70", it shouldn't be removed because it contains a letter.
  """
  # Regular expression to match strings containing only digits or special characters
  pattern = r'^[\d_]+$'

  # Filter out strings that match the pattern
  filtered_arr = [s for s in arr if not re.match(pattern, s)]

  return filtered_arr

def remove_empty_strings(array):
    """
    Filter out empty strings
    """
    filtered_array = [s for s in array if s.strip() != '']

    return filtered_array

def clean_array(array):
    """
    Uses the two functions above to remove strings with only special and numerical characters
    and empty strings.
    """

    # Removes special and numeric strings
    array = remove_special_and_numeric_strings(array)

    # Removes empty spaces
    array = remove_empty_strings(array)
    return array

def filter_long_strings(array, max_len):
    """
    Removes any string longer than a certain threshold given by max_len
    """

    filtered_arr = [s for s in array if len(s) <= max_len]
    return filtered_arr

def filter_short_strings(array,min_len):
    """
    Removes any string shorter than a certain threshold given by min_len
    """

    filtered_arr = [s for s in array if len(s) > min_len]
    return filtered_arr

def non_stop(array):
  """
  Removes any stop words in the array
  """
  stop_words = set(stopwords.words('english'))
  new_array = []

  for string in array:
    word_tokens = word_tokenize(string)

    for w in word_tokens:
      if w not in stop_words:
          new_array.append(w)
  return new_array

def lower_case(array):
  """
  This function will return every string in lower case
  """

  new_array = [string.lower() for string in array]
  return new_array

def lemmatize_word(word, pos_tag):
    """
    Lemmatize word based on its POS tag.
    """
    if pos_tag.startswith('J'):
        return WordNetLemmatizer().lemmatize(word, wordnet.ADJ)
    elif pos_tag.startswith('V'):
        return WordNetLemmatizer().lemmatize(word, wordnet.VERB)
    elif pos_tag.startswith('N'):
        return WordNetLemmatizer().lemmatize(word, wordnet.NOUN)
    elif pos_tag.startswith('R'):
        return WordNetLemmatizer().lemmatize(word, wordnet.ADV)
    else:
        return WordNetLemmatizer().lemmatize(word)

def lemmatize_sentence(sentence):
    """
    Lemmatize each word in the sentence.
    """
    # Tokenize the sentence into words
    words = nltk.word_tokenize(sentence)
    # Get the POS tags for each word
    pos_tags = nltk.pos_tag(words)
    # Lemmatize each word with its corresponding POS tag
    lemmatized_sentence = ' '.join([lemmatize_word(word, pos_tag) for word, pos_tag in pos_tags])
    return lemmatized_sentence
    

def lemmatize_term(term):
    """
    Lemmatize a single term.
    """
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(term, pos='n')  # Assuming all terms are nouns

def lemmatize_list(terms_list):
    """
    Lemmatize a list of terms.
    """
    return [lemmatize_term(term) for term in terms_list]


def clean(input_array):
  output_array = []
  for lst in input_array:
    new_lst = separate_strings(lst)
    new_lst = replace_strings(new_lst)
    new_lst = clean_array(new_lst)
    new_lst = [lemmatize_sentence(sentence) for sentence in new_lst]
    new_lst = non_stop(new_lst)
    new_lst = filter_short_strings(new_lst,2)
    output_array.append(new_lst)
    

  return output_array

# Function that removes duplicate keywords for each article

def remove_duplicate_keywords(input_array):
  output_array = []
  for article in input_array:
    local_array = []
    for kw in article:
      if kw not in local_array:
        local_array.append(kw)
    output_array.append(local_array)
  return output_array





################################################



def embeddings(input_array1,input_array2,input_array3):
    # input_array1 is a list of abstacts
    # input_array2 is a list of keywords
    # input_array3 is a list of entities

    documents = input_array1

    # Each nested array represents each abstract's article
    documents_list = [[lst.lower()] for lst in documents]


    # In this case a sample is the whole dataset's Abstract column
    sample = documents_list

    # All text data pre processed
    sentences_clean = clean(sample)


    # Train the Word2Vec model on the articles' abstracts
    model_word_embeddings = Word2Vec(sentences=sentences_clean, min_count=1, vector_size=100)

    # Save the model
    model_word_embeddings.save("model_word_embeddings.bin")


    # - vector_size: The dimensionality of the word vectors.
    # - window: The maximum distance between the current and predicted word within a sentence.
    # - min_count: Ignores all words with a total frequency lower than this.
    # - workers: The number of threads to use while training the model.

    # Create the embeddings for each word in the vocabulary
    vocabulary_embeddings = {string: model_word_embeddings.wv[string] for sentence in sentences_clean for string in sentence}

    # Calculate each words' weight
    # Number of times a word appears in the whole text data
    word_weights= {}

    for document in sentences_clean:
        for string in document:
            if string in word_weights:
                word_weights[string] += 1
            else:
                word_weights[string] = 1
    

    # Next, we clean all keywords data
    output_array = []
    for doc in input_array2:
        doc = lower_case(doc)
        new_doc = [string.replace(":",",").lower() for string in doc]
        new_doc = [string.replace("and","").replace(":",",").replace(" - ",", ").replace("-","").replace("'","") for string in doc]
        new_lst = [string.split(",") for string in new_doc]
        new_doc = [string for nested_array in new_lst for string in nested_array]
        output_array.append(new_doc)
    
    #############################################################################################
    
    # Embeddings
    # Creates data structures to save keywords' embeddings and the list of keywords per article

    multi_kw_vectors_weighted = {}
    lst_keywords_docs_weighted = []

    for doc in output_array:
        local_lst_keywords_doc = []
        for kw in doc:

            parts = kw.split(" ")
            lem_parts = lemmatize_list(parts)

            list_kw_weight = []
            for i in range(0,len(lem_parts)):
                if lem_parts[i] in vocabulary_embeddings.keys():
                    list_kw_weight.append(lem_parts[i])


            if len(list_kw_weight) >1:
                string = list_kw_weight[0]
                total = word_weights[list_kw_weight[0]]
                weighted_mean = vocabulary_embeddings[list_kw_weight[0]]*word_weights[list_kw_weight[0]]
                if len(list_kw_weight) == 1:
                    continue
                else:
                    for i in range(0,len(list_kw_weight)-1):
                        string = string + " " + list_kw_weight[i+1]
                        total += total + word_weights[list_kw_weight[i+1]]
                        weighted_mean += vocabulary_embeddings[list_kw_weight[i+1]]*word_weights[list_kw_weight[i+1]]
                multi_kw_vectors_weighted[string] = weighted_mean/total
            else:
                continue
            local_lst_keywords_doc.append(string)
        lst_keywords_docs_weighted.append(local_lst_keywords_doc)

    #########################################################

    # Creates a data structure to store entities' embeddings

    multi_e_vectors_weighted = {}

    mapped_e = {}
    for entity in input_array3:
        parts = entity.split(" ")
        lem_parts = lemmatize_list(parts)

        # The strategic value of string is to ensure that even entities that do not appear in the vocabulary embeddings, will be taken into account
        string=""

        list_e_weight = []
        for i in range(0,len(lem_parts)):
            if lem_parts[i] in vocabulary_embeddings.keys():
                list_e_weight.append(lem_parts[i])


        if len(list_e_weight) > 1:
            string = list_e_weight[0]
            total = word_weights[list_e_weight[0]]
            weighted_mean = vocabulary_embeddings[list_e_weight[0]]*word_weights[list_e_weight[0]]
            if len(list_e_weight) == 1:
                continue
            else:
                for i in range(0,len(list_e_weight)-1):
                    string = string + " " + list_e_weight[i+1]
                    total += total + word_weights[list_e_weight[i+1]]
                    weighted_mean += vocabulary_embeddings[list_e_weight[i+1]]*word_weights[list_e_weight[i+1]]
            multi_e_vectors_weighted[string] = weighted_mean/total
            
        else:
            continue

        mapped_e[entity]=string
            
    entities_weighted_list = list(multi_e_vectors_weighted.keys())

    return lst_keywords_docs_weighted,multi_kw_vectors_weighted,entities_weighted_list,multi_e_vectors_weighted,mapped_e


def alignment_with_weights(array1,dict1,array2,dict2,theta):

    new_list_kw = []
    for doc in array1:
        # each doc contains a nested array of keywords
        new_local_kw = []

        for kw in doc:
            if kw != "":
                # similarity values between a keyword and each entity
                similarity_matrix = [cosine_similarity(dict1[kw].reshape(1,-1),dict2[e].reshape(1,-1))[0][0] for e in array2]
                # find the max value of this list
                max_s = max(similarity_matrix)
                # Find the index of the maximum value
                max_index = similarity_matrix.index(max_s)
                if max_s >= theta:
                    new_local_kw.append(array2[max_index])

        new_list_kw.append(new_local_kw)

    return new_list_kw


def main(input_array1,input_array2,input_array3,threshold_value,data_file1 = None,data_file2=None):
    if data_file1 == None and data_file2 == None:
       
        lst_keywords_docs_weighted,multi_kw_vectors_weighted,entities_weighted_list,multi_e_vectors_weighted,map_entities = embeddings(input_array1,input_array2,input_array3)
 
        list_keywords = alignment_with_weights(lst_keywords_docs_weighted,multi_kw_vectors_weighted,entities_weighted_list,multi_e_vectors_weighted,threshold_value)
    
    else:
       list_keywords = []
       map_entities = {}
    
       with open(data_file2, 'r') as file:
            # Load JSON data from the file into a Python dictionary or list
            map_entities = json.load(file)

       # Open the file in read mode
       with open(data_file1, "r") as file:
            # Read each line of the file
            for line in file:
                # Safely evaluate the string as a Python literal expression
                nested_list = ast.literal_eval(line.strip())
                # Append the nested list to the data list
                list_keywords.append(nested_list)

        

    data_kw_filtered = remove_duplicate_keywords(list_keywords)
    for index,d in enumerate(data_kw_filtered):
        if len(d) == 0:
            data_kw_filtered[index] = ["Empty"]
        else:
           continue

    # Procedure to count the percentage of Empty lists in the data
    print("Number of entries: ", len(data_kw_filtered))
    n_empty = 0
    for article in data_kw_filtered:
        for keyword in article:
            if keyword == "Empty":
                n_empty += 1
    print("Number of empty entries: ", n_empty)
    print("It corresponds to ",round(n_empty/len(data_kw_filtered)*100,2),"%")

       
    return data_kw_filtered,map_entities


        