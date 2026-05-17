"""
Caches expensive Pytorch tensors to files.

Use the higher order function cache_after_first_run

Example usage:

cache_after_first_run(load_words_as_tensor, 'words')
cache_after_first_run(lambda : load_words_as_tensor(size=12), 'words12')
"""

import os

import torch
from xdg_base_dirs import xdg_cache_home

def get_cache_path(filename):
    "Retrieves the cache path."
    cache_dir = os.path.join(xdg_cache_home(), 'nnfont/')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, filename)

def write_cache(data, filename):
    "Writes a data to a cache of the given filename."
    torch.save(data, get_cache_path(filename))

def read_cache(filename):
    "Reads data from a cache of the given filename."
    return torch.load(get_cache_path(filename))

def cache_exists(filename):
    "Checks to see if the cache file already exists."
    return os.path.isfile(get_cache_path(filename))

def cache_after_first_run(function, filename):
    "Caches a file after the first time it runs."
    if cache_exists(filename):
        return read_cache(filename)

    data = function()
    write_cache(data, filename)
    return data
