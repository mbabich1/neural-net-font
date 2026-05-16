import re

import freetype
import torch
from torch import tensor

def load_face(path, size):
    "Loads the requested face at the requested size."
    face = freetype.Face(path)
    face.set_char_size(size * 64)
    return face

def kerning_value(face, previous_char, char):
    "Turns the face kerning into the kerning value."
    if previous_char:
        return face.get_kerning(previous_char, char).x >> 6
    return 0

def advance_x(glyph):
    "Advances the x coordinate of the glyph."
    return glyph.advance.x >> 6

def determine_text_dimensions(text, face, size):
    """
    Calculate the size of the smallest 2D tensor that contains the
    given text.
    """
    top = -size
    bottom = size
    width = 0
    glyph = face.glyph
    previous_char = None
    # Walk through each character. If the top is larger or the bottom
    # is smaller than the old top or bottom, then update that to
    # reflect the new dimensions. Then advance the width by the
    # specified x advancement adjusted by the kerning value.
    for char in text:
        face.load_char(char)
        top = max(top, glyph.bitmap_top)
        bottom = min(bottom, glyph.bitmap_top - glyph.bitmap.rows)
        width += advance_x(glyph) + kerning_value(face, previous_char, char)
        previous_char = char
    return top, (top - bottom), width

def render_text(data, text, face, top):
    "Draw each glyph of the text on top of an empty data tensor."
    x = 0
    glyph = face.glyph
    previous_char = None
    # Walk through each character. Write the glyph data on top of the
    # empty data tensor.
    for char in text:
        face.load_char(char)
        x += kerning_value(face, previous_char, char)
        y = top - glyph.bitmap_top
        left = glyph.bitmap_left
        width = glyph.bitmap.width
        height = glyph.bitmap.rows
        glyph_tensor = tensor(glyph.bitmap.buffer, dtype=torch.uint8).reshape(height, width)
        # Write over the data from the (x + left, y) corner to the
        # (x + left + width, y + height) corner with the glyph tensor.
        data[y : y + height, x + left : x + left + width] += glyph_tensor
        x += advance_x(glyph)
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

def create_text_data(text, face, size):
    "Creates a tensor that fits the text, face, and size."
    top, height, width = determine_text_dimensions(text, face, size)
    print(width)
    data = torch.zeros((height, width), dtype=torch.uint8)
    render_text(data, text, face, top)
    print_font_data(data)

# Turns a glyph into an array that can be turned into a tensor for
# pytorch
def main():
    text = "The quick brown fox jumps over the lazy dog."
    texts = split_text(text)
    # TODO: fixme... why does it work on the text, but not on either
    # of the split texts?
    print(texts)
    # TODO: include the two fonts with the repo so it works on any distro/OS?
    # TODO: also serif
    font_path = '/usr/share/fonts/google-noto/NotoSans-Regular.ttf'
    size = 12
    face = load_face(font_path, size)
    create_text_data(text, face, size)

if __name__ == "__main__":
    main()
