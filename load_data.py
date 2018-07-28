from bs4 import BeautifulSoup
from keras import preprocessing as pp
import pandas as pd
import json


# Constants
max_features = 100000
languages = ('angular', 'asm', 'asp.net', 'c#', 'c++', 'css', 'delphi', 'html',
        'java', 'javascript', 'objectivec', 'pascal', 'perl', 'php',
        'powershell', 'python', 'razor', 'react', 'ruby', 'scala', 'sql',
        'swift', 'typescript', 'vb.net', 'xml')


def get_code_labels(filename="new_languages.txt", resort_to_fallback=False):
    soup = None
    try:
        soup = BeautifulSoup(open(filename), "html.parser")

    except FileNotFoundError as fnf:
        if resort_to_fallback:
            print("input file {} not found, resorting to fallback file 'languages.txt'".format(filename))
            soup = BeautifulSoup(open("languages.txt"), "html.parser")
        else:
            print("input file {} not found, returning None, None".format(filename))
            return None, None
    finally:
        # create empty lists for the samples and the labels
        code_samples, labels = list(), list()

        # iterate through each pretag and extract the code and append it to the sample list
        for tag in soup.find_all(name="pre", text=True):
            code_samples.append(str(tag.contents[0]))
            labels.append(tag["lang"].lower())

        return code_samples, labels


# Create a dataset from a given filename
def create_tokenized_data_set(json_file="data_index.json", overwrite_file=False, features=max_features):
    code, labels = get_code_labels(filename="new_languages.txt")

    # Create a tokenizer to preprocess our text data
    tokenizer = pp.text.Tokenizer(num_words=features)
    # This will take the text that we input from the given datafile and index every unique word and character
    tokenizer.fit_on_texts(code)
    tokenized_word_index = tokenizer.word_index
    try:
        # try to open the file in no-overwrite mode
        with open(file=json_file, mode="x") as outfile:
            # Dump the tokenized index to a json file externally
            json.dump(obj=tokenized_word_index, fp=outfile)
    # If the file already exists
    except FileExistsError as fee:
        print("Found existing {} file".format(json_file))
        if overwrite_file:
            print("overwriting")
            with open(file=json_file, mode='w') as outfile:
                json.dump(obj=tokenized_word_index, fp=outfile)

    except Exception as e:
        print("Exception was caused by {} within {}".format(e.__cause__, e.__traceback__))

    X = tokenizer.texts_to_sequences(code)
    # Pad the input sequences to be 100 input sequences at most
    return pp.sequence.pad_sequences(sequences=X, maxlen=100), pd.get_dummies(data=labels)


# Given a stream of text, convert it into an index array
def text_to_index_array(text):
    word_vector = []
    with open(file="data_index.json", mode='r') as infile:
        dictionary = json.load(infile)

    print("json file: {}".format(json.dumps(dictionary, indent=4)))
    for word in pp.text.text_to_word_sequence(text):
        if word in dictionary:
            if dictionary[word] <= max_features:
                word_vector.append([dictionary[word]])
            else:
                word_vector.append([0])  # append just a 0 to denote an undefined/unseen variable
        else:
            word_vector.append([0])  # append a zero to denote an <unknown> word feature

    return word_vector


# Creates the necessary input set needed for code evaluation
def create_input_evaluation(filename):
    raw_code = ""
    try:
        with open(file=filename, mode='r') as readfile:
            raw_code += readfile.read()
    except FileNotFoundError:
        print("{} was not found, returning.".format(filename))
        return None

    # convert the raw code given into an indexed array
    indexed_text = text_to_index_array(raw_code)
    return pp.sequence.pad_sequences(sequences=list(indexed_text), maxlen=100)  # Pad the sequence to be at most 100


