"""
Loads the data set from Kaggle.

This can be easily extended to load other data sets from Kaggle.
"""

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

def process_words_dataset():
    """
    Returns the Pandas-processed words dataset, which has two columns:
    'word' and 'pos_tag', where the latter is the parts of speech tag.
    This second column, the data label, is not currently used.
    """
    return pd.read_csv(get_words_dataset())
