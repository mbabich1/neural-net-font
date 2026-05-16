import pandas as pd
import kagglehub

def get_words_dataset():
    """
    Returns the path to the words dataset.

    Note that this dataset is a CSV.

    Downloads the words dataset if it is not already there.
    """
    path = kagglehub.dataset_download("ruchi798/part-of-speech-tagging")
    return path + '/words_pos.csv'

# TODO: Not handled yet. Pandas only does easy CSVs.
def get_sentences_dataset():
    """
    Returns the path to the sentences dataset.

    Note that this dataset is line-by-line.

    Downloads the sentences dataset if it is not already there.
    """
    path = kagglehub.dataset_download("mikeortman/wikipedia-sentences")
    return path + '/wikisent2.txt'

def process_words_dataset():
    return pd.read_csv(get_words_dataset())['word']

# print(process_words_dataset()[0])
