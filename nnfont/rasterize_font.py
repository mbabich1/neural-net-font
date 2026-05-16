import freetype

from math import floor

# Turns a glyph into an array that can be turned into a tensor for
# pytorch
def main():
    # TODO: include the two fonts with the repo so it works on any distro/OS?
    face = freetype.Face('/usr/share/fonts/google-noto/NotoSans-Regular.ttf')
    # TODO: should this size be hardcoded?
    face.set_char_size(48*64)
    face.load_char('A')
    bitmap = face.glyph.bitmap
    for i in range(floor(len(bitmap.buffer) / bitmap.width)):
        row = bitmap.buffer[bitmap.width*i:bitmap.width*(i+1)]
        print(row)
    # TODO: use kerning/etc. glyph information to turn a string into
    # an array instead of just one character

if __name__ == "__main__":
    main()
