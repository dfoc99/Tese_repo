import os
import pandas as pd
import re



def read_file(path_file):
    """
        This function's goal is to read a file
    """
    with open(path_file, 'r') as file:
        # Read the contents of the file
        file_contents = file.readlines()
        lines = [line.strip() for line in file_contents]

        return lines


# Funtion that assign each article to a key
def process_list(input_list):

    """
        Followed by the function read_file, this function separates each block of information
        into entries. This only works due to the BibText formatation.
    """
    result_dict = {}
    current_value = ""
    key_counter = 0

    for item in input_list:
        if item == '':
            result_dict[key_counter] = current_value
            key_counter += 1
            current_value = ""

        else:
            current_value = current_value + ' ' + item


    return result_dict


def extract_article_info(dataset):
    """ This function finds specific information, through the use of Regex Expressions, about each article such as:
          - The name of the article;
          - The authors' names;
          - Articles' Keywords;
          - Articles' Abstracts;
          - Articles' Journals;
          - The year each article is published.

        Afterwards, it stores each entry to a dataframe
    """
    articles_info = []

    for item in dataset:
        info = {}

        # Extracting relevant information using regex
        match_title = re.search(r'title\s*=\s*"{([^"]+)}"', dataset[item], re.IGNORECASE)
        match_authors = re.search(r'author\s*=\s*\{([^\d.]*)},title', dataset[item], re.IGNORECASE)
        match_keywords = re.search(r'keywords\s*=\s*{([^}]+)}', dataset[item], re.IGNORECASE)
        match_abstract = re.search(r'abstract\s*=\s*"{([^"]+)}"', dataset[item], re.IGNORECASE)
        match_journal = re.search(r'(journal|booktitle)\s*=\s*{([^}]+)}', dataset[item], re.IGNORECASE)
        match_year = re.search(r'year\s*=\s*(\d+)', dataset[item], re.IGNORECASE)

        # Populate the dictionary with matched values
        if match_title:
            info['Title'] = match_title.group(1)
        if match_authors:
            info['Authors'] = match_authors.group(1)
        if match_keywords:
            info['Keywords'] = match_keywords.group(1)
        if match_abstract:
            info['Abstract'] = match_abstract.group(1)
        if match_journal:
            info['Journal'] = match_journal.group(2)
        if match_year:
            info['Year'] = int(match_year.group(1))

        articles_info.append(info)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(articles_info)
    return df



def main(file):
    # Reads the .txt file
    file_articles = read_file(file)

    # Separates each block as an article
    processed_articles = process_list(file_articles)
    
    # Stores each description into a dataframe
    data_articles = extract_article_info(processed_articles)

    # Converts float values of the column "Year" into int values
    data_articles['Year'] = pd.to_numeric(data_articles['Year'], errors='coerce').astype('Int64')
    
    # Drops NaN values of the Abstract column
    data_articles = data_articles.dropna(subset=['Abstract'])

    # Drops duplicate values
    data_articles = data_articles.drop_duplicates()

    # Count the number of NaN values in 'column1'
    nan_count = data_articles['Abstract'].isna().sum()

    print("Number of NaN values in 'Abstract':", nan_count)

    return data_articles


