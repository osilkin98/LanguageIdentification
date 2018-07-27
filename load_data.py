from bs4 import BeautifulSoup
from keras import preprocessing as pp
import pandas as pd
import json


def get_code_labels(filename="filename.txt"):
    soup = BeautifulSoup(open(filename), "html.parser")

    # create empty lists for the samples and the labels
    code_samples, labels = list(), list()

    # iterate through each pretag and extract the code and append it to the sample list
    for tag in soup.find_all(name="pre", text=True):
        code_samples.append(str(tag.contents[0]))
        labels.append(tag["lang"].lower())

    return code_samples, labels


# Create a dataset from a given filename
def create_tokenized_data_set(json_file="data_index.json", overwrite_file=False, features=10000):
    code, labels = get_code_labels()

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
            with open(file=json_file, mode='x') as outfile:
                json.dump(obj=tokenized_word_index, fp=outfile)

    except Exception as e:
        print("Exception was caused by {} within {}".format(e.__cause__, e.__traceback__))

    X = tokenizer.texts_to_sequences(code)
    # Pad the input sequences to be 100 input sequences at most
    return pp.sequence.pad_sequences(sequences=X, maxlen=100), pd.get_dummies(data=labels)