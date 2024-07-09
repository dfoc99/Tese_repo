# Read the ontology file 
 
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



def read_jsonFile(file_path):
  # Open the JSON file and load its contents as a dictionary
  with open(file_path, 'r') as file:
    data = json.load(file)

  return data


# Function to check if a string is a URI
def is_uri(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def ontology(file_path):

    triples = read_jsonFile(file_path)
    filtered_triples = {id_: triple for id_, triple in triples.items() if not any(is_uri(obj) for obj in triple)}
    print("Size of triples data: ",len(triples))

    # List of entities derived from the ontology
    list_entities = []
    for i in filtered_triples:
        entity1 = triples[i][0]
        entity2 = triples[i][2]
        if entity1 in list_entities and entity2 not in list_entities:
            list_entities.append(entity2)
        elif entity1 not in list_entities and entity2 in list_entities:
            list_entities.append(entity1)
        elif entity1 not in list_entities and entity2 not in list_entities:
            list_entities.append(entity1)
            list_entities.append(entity2)
        else:
            continue
    
    print("Size of list of entitites: ",len(list_entities))

    # List of unique entities, meaning that it is a list that contains every entity that appears in the whole ontology 
    unique_entities = []
    for i in list_entities:
        if i not in unique_entities:
            unique_entities.append(i)

    multi_entities = []

    for entity in unique_entities:
        entity = entity.replace(":","_")
        entity = entity.replace(" ","_")
        multi_entities.append(entity)

    for i in range(len(multi_entities)):
        # Check if the string contains a double underscore
        if "__" in multi_entities[i]:
            # Replace the second underscore with an empty space
            multi_entities[i] = multi_entities[i].replace("__", " ", 1)
        elif "_-_" in multi_entities[i]:
            multi_entities[i] = multi_entities[i].replace("_-_", " ", 1)

    for i in range(len(multi_entities)):
        # Check if the string contains a double underscore
        if "_" in multi_entities[i]:
            # Replace the second underscore with an empty space
            multi_entities[i] = multi_entities[i].replace("_", "-")

    # list of the relations within the triples information
    list_relations = []
    for i in triples:
        relation = triples[i][1]
        if relation not in list_relations:
            list_relations.append(relation)
        list_relations



    ###########################

    # entities to be removed 
    entities_removed = []

    # Descriptions
    dict_definition = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == 'definition':
            dict_definition[entity1] = entity2
    dict_definition_reversed =  {value: key for key, value in dict_definition.items()}
    entities_removed.append(list(dict_definition_reversed.keys()))


    # Comment
    dict_comment = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == 'comment':
            dict_comment[entity1] = entity2
    dict_comment_reversed =  {value: key for key, value in dict_comment.items()}
    entities_removed.append(list(dict_comment.keys()))
    entities_removed.append(list(dict_comment_reversed.keys()))


    # Concepts in both american and uk english
    dict_editorialNote = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == "editorialNote":
            if entity1 != entity2:
                dict_editorialNote[entity1] = entity2
    dict_editorialNote_reversed =  {value: key for key, value in dict_editorialNote.items()}
    entities_removed.append(list(dict_editorialNote_reversed.keys()))


    # Titles
    dict_title = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == 'title':
            dict_title[entity1] = entity2
    entities_removed.append(list(dict_title.keys()))


    # ScopeNote
    dict_scopeNote = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == "scopeNote":
            if entity1 != entity2:
                dict_scopeNote[entity1] = entity2
    dict_scopeNote_reversed = {value: key for key,value in dict_scopeNote.items()}
    entities_removed.append(list(dict_scopeNote_reversed.keys()))


    # ChangeNote
    dict_changeNote = {}
    for i in triples:
        entity1 = triples[i][0]
        relation = triples[i][1]
        entity2 = triples[i][2]
        if relation == 'changeNote':
            dict_changeNote[entity1] = entity2
    dict_changeNote_reversed = {value:key for key,value in dict_changeNote.items()}
    entities_removed.append(list(dict_changeNote_reversed.keys()))


    # List of the unwanted entities
    final_entities_removed = [entity for nested_array in entities_removed for entity in nested_array ]

    # Procedure to remove unwanted entities
    unique_entities_filtered = []
    count = 0
    for entity in unique_entities:
        if entity not in final_entities_removed:
            unique_entities_filtered.append(entity)
        else:
            count = count + 1

    print("Number of remaining filtered entities: ", len(unique_entities_filtered))

    return unique_entities_filtered


def main(file_path):
    list_entities = ontology(file_path=file_path)
    # input_array3

    return list_entities
