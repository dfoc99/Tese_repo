import json

def read_file(file_path):

    # Read the contents of the file
    with open(file_path, 'r') as file:
        json_data = file.read()

    # Load JSON data into a Python dictionary
    dictionary = json.loads(json_data)

    return dictionary

def keywords_counter(dictionary):
    # Extracts each key of the dictionary which is the abstract
    abstract_list = list(dictionary.keys())

    Keywords_counter = {}
    for i in abstract_list:
        for j in dictionary[i]:
            if j[0] in Keywords_counter:
                # j[0] is the keyword of the nested list
                Keywords_counter[j[0]] += 1
            else:
                Keywords_counter[j[0]] = 1
    return Keywords_counter

def keywords_max_count(counter):
    c=0
    for i in counter:
        if counter[i] > c:
            c = counter[i]
            d = c, i
        else:
            continue
    return d


def main(data,file = None):

    if file == None:

        from keybert import KeyBERT

        kw_model = KeyBERT()

        # To be able to see results, we are going to select only 5 articles, to check how well extraction works

        

        extraction_KeyBERT = {}

        data_abstract = data["Abstract"]

        for i in data_abstract:
            extraction_KeyBERT[i] = kw_model.extract_keywords(i,keyphrase_ngram_range=(1, 1), top_n=5)
            extraction_KeyBERT[i].extend(kw_model.extract_keywords(i,keyphrase_ngram_range=(1, 2), top_n=5))
            extraction_KeyBERT[i].extend(kw_model.extract_keywords(i,keyphrase_ngram_range=(3, 3), top_n=5))     
    else:
        extraction_KeyBERT = read_file(file)

    data_extracted = extraction_KeyBERT
    n = keywords_counter(extraction_KeyBERT)
    m = keywords_max_count(n) 


    return n,m,data_extracted





