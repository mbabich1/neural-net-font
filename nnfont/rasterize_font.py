"""
Loads the data into a tensor for processing.

The main entry point is load_words_as_tensor

The function flatten_words can flatten that tensor's word data to 1D
"""

from math import log, ceil
import re

import freetype
import torch
from torch import tensor

from nnfont.data import process_words_dataset

CACHE = {}
GLYPH_TENSORS = {}
KERNING_CACHE = {}

def load_face(path, size):
    "Loads the requested face at the requested size."
    face = freetype.Face(path)
    face.set_char_size(size * 64)
    return face

def kerning_value(face, previous_char, char):
    "Turns the face kerning into the kerning value."
    if previous_char:
        pair = previous_char + char
        if pair in KERNING_CACHE:
            return KERNING_CACHE[pair]
        kerning = face.get_kerning(previous_char, char).x >> 6
        KERNING_CACHE[pair] = kerning
        return kerning
    return 0

def advance_x(glyph):
    "Advances the x coordinate of the glyph."
    return glyph.advance.x >> 6

def glyph_data(face, char):
    "Memoize the glyph data into a cache because it's very repetitive."
    if char in CACHE:
        return CACHE[char]
    face.load_char(char)
    glyph = face.glyph
    top = glyph.bitmap_top
    height = glyph.bitmap.rows
    width = glyph.bitmap.width
    left = glyph.bitmap_left
    adv_x = advance_x(glyph)
    result = (top, height, width, left, adv_x)
    CACHE[char] = result
    GLYPH_TENSORS[char] = tensor(glyph.bitmap.buffer, dtype=torch.uint8).reshape(height, width)
    return result

def determine_text_dimensions(text, face, size):
    """
    Calculate the size of the smallest 2D tensor that contains the
    given text.
    """
    top = -size
    bottom = size
    width = 0
    previous_char = None
    # Walk through each character. If the top is larger or the bottom
    # is smaller than the old top or bottom, then update that to
    # reflect the new dimensions. Then advance the width by the
    # specified x advancement adjusted by the kerning value.
    for char in text:
        glyph_top, glyph_height, glyph_width, left, adv_x = glyph_data(face, char)
        top = max(top, glyph_top)
        bottom = min(bottom, glyph_top - glyph_height)
        # The first character can start negative, e.g. "j"
        if left < 0 and width == 0:
            width += abs(left)
        width += adv_x + kerning_value(face, previous_char, char)
        previous_char = char
    # Note: Adding a padding of one to the width can fix off-by-one
    # errors at the end of the text.
    return top, (top - bottom), width + 1

# TODO: center the text?
def render_text(data, text, face, top, i = None):
    "Draw each glyph of the text on top of an empty data tensor."
    x = 0
    glyph = face.glyph
    previous_char = None
    # Walk through each character. Write the glyph data on top of the
    # empty data tensor.
    for char in text:
        x += kerning_value(face, previous_char, char)
        glyph_top, height, width, left, adv_x = glyph_data(face, char)
        y = top - glyph_top
        glyph_tensor = GLYPH_TENSORS[char]
        # The first character can start negative, e.g. "j"
        if left < 0 and x == 0:
            x += abs(left)
        # Write over the data from the (x + left, y) corner to the
        # (x + left + width, y + height) corner with the glyph tensor.
        if i is None:
            data[y : y + height, x + left : x + left + width] += glyph_tensor
        else:
            data[i, y : y + height, x + left : x + left + width] += glyph_tensor
        x += adv_x
        previous_char = char

def print_font_data(data):
    "For debugging, prints all of the rendered text."
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print('{:4d}'.format(int(data[i][j])), end='')
        print()

def split_text(text):
    "Add a line break in the middle of a sentence."
    mid = len(text) // 2
    # Split on the nearest whitespace to the right of mid
    split1 = re.split(r'(\s+)', text[mid:], maxsplit=1)
    l1 = text[:mid] + split1[0]
    r1 = split1[2]
    # Split on the nearest whitespace to the left of mid
    split2 = re.split(r'(\s+)', text[:mid][::-1], maxsplit=1)
    l2 = split2[2][::-1]
    r2 = split2[0][::-1] + text[mid:]
    # Prefer whichever split is closer to the middle
    if abs(len(l1) - len(r1)) <= abs(len(l2) - len(r2)):
        return (l1, r1)
    return (l2, r2)

def create_one_line_text_data(text, face, size, array = None, i = None):
    "Creates a tensor that fits one line of text."
    top, height, width = determine_text_dimensions(text, face, size)
    if tensor is None:
        data = torch.zeros((height, width), dtype=torch.uint8)
    else:
        data = array
    render_text(data, text, face, top, i = i)
    return data

# This is the main API function of this file right now.
def create_multiline_text_data(text, face, size):
    "Creates a tensor that fits the multiline text."
    texts = split_text(text)
    a = create_one_line_text_data(texts[0], face, size)
    b = create_one_line_text_data(texts[1], face, size)
    # The break between the line
    y_offset = size // 6
    data = torch.zeros((a.shape[0] + b.shape[0] + y_offset,
                        max(a.shape[1], b.shape[1])),
                       dtype=torch.uint8)
    y_start = 0
    data[y_start : y_start + a.shape[0], 0 : a.shape[1]] += a
    y_start += a.shape[0] + y_offset
    data[y_start : y_start + b.shape[0], 0 : b.shape[1]] += b
    return data

def create_text_data(text, face, size, array = None, i = None):
    "Creates a tensor that fits the text, face, and size."
    if len(text) >= 20 and re.search(r'(\s+)', text):
        return create_multiline_text_data(text, face, size)
    return create_one_line_text_data(text, face, size, array = array, i = i)

def determine_array_size(data, face, size):
    "Do a first pass through the data to find how big to make the tensor."
    x_max = -1
    y_max = -1
    for text in data:
        top, height, width = determine_text_dimensions(text, face, size)
        shape = (height, width)
        x_max = max(shape[1], x_max)
        y_max = max(shape[0], y_max)
    x_max = 2 ** ceil(log(x_max, 2))
    y_max = 2 ** ceil(log(y_max, 2))
    return len(data), y_max, x_max

def fill_array(array, data, face, size):
    "Fills the array with the rasterized data."
    for i, text in enumerate(data):
        create_one_line_text_data(text, face, size, array = array, i = i)
    return array

def load_words_as_tensor(font_path = 'fonts/NotoSans-Regular.ttf', size = 8, n = None):
    "Rasterizes all of the word data set with the given font and font size."
    # Reset the global cache dicts
    CACHE.clear()
    GLYPH_TENSORS.clear()
    KERNING_CACHE.clear()
    # Load the data
    face = load_face(font_path, size)
    words = process_words_dataset()['word']
    if n is not None:
        words = words[:n]
    # Create and populate the tensor array.
    array = torch.zeros(determine_array_size(words, face, size),
                        dtype=torch.uint8)
    fill_array(array, words, face, size)
    return array

def flatten_words(words):
    "Flatten the words from 2D to 1D, flattening the tensor from 3D to 2D."
    return torch.reshape(words, (words.shape[0], words.shape[1] * words.shape[2]))

# Turns a glyph into an array that can be turned into a tensor for
# pytorch... when not called directly, this serves as a model for how
# to use the API.
def main():
    "Test print one of the items when called directly."
    data = load_words_as_tensor()
    # Print the middle word's array data
    print_font_data(data[data.shape[0] // 2])
    print(flatten_words(data).shape)

if __name__ == "__main__":
    main()
