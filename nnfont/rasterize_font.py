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

def render_text(data, text, face, top):
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
        data[y : y + height, x + left : x + left + width] += glyph_tensor
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

def create_one_line_text_data(text, face, size):
    "Creates a tensor that fits one line of text."
    top, height, width = determine_text_dimensions(text, face, size)
    data = torch.zeros((height, width), dtype=torch.uint8)
    render_text(data, text, face, top)
    return data

# This is the main API function of this file right now.
#
# TODO: pad them all out to the same dimensions?
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

def create_text_data(text, face, size):
    "Creates a tensor that fits the text, face, and size."
    if len(text) >= 20 and re.search(r'(\s+)', text):
        return create_multiline_text_data(text, face, size)
    return create_one_line_text_data(text, face, size)

# Turns a glyph into an array that can be turned into a tensor for
# pytorch... when not called directly, this serves as a model for how
# to use the API.
def main():
    # Note: The caches have to be cleared if the font changes.
    font_path = 'fonts/NotoSans-Regular.ttf'
    size = 12
    face = load_face(font_path, size)
    words = process_words_dataset()['word']
    # text = "The quick brown fox jumps over the lazy dog."
    x_max = -1
    y_max = -1
    for text in words[:len(words)]:
        shape = create_text_data(text, face, size).shape
        x_max = max(shape[1], x_max)
        y_max = max(shape[0], y_max)
    print(x_max, y_max)

if __name__ == "__main__":
    main()
