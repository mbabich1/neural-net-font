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
        # Write over the data from the (x + left, y) corner to the (x
        # + left + width, y + height) corner with the glyph tensor.
        data[y : y + height, x + left : x + left + width] += glyph_tensor
        x += advance_x(glyph)
        previous_char = char

def print_font_data(data):
    "For debugging, prints all of the rendered text."
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print('{:4d}'.format(int(data[i][j])), end='')
        print()

# Turns a glyph into an array that can be turned into a tensor for
# pytorch
def main():
    text = "Hello, world!"
    # TODO: include the two fonts with the repo so it works on any distro/OS?
    font_path = '/usr/share/fonts/google-noto/NotoSans-Regular.ttf'
    size = 12
    face = load_face(font_path, size)
    top, height, width = determine_text_dimensions(text, face, size)
    data = torch.zeros((height, width), dtype=torch.uint8)
    render_text(data, text, face, top)
    print_font_data(data)

if __name__ == "__main__":
    main()
