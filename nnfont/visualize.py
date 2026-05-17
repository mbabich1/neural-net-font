import matplotlib.pyplot as plt
import rasterize_font

def plot_word(words, word_id):
    plt.style.use('_mpl-gallery-nogrid')
    word = words[word_id]
    fig, ax = plt.subplots()
    ax.imshow(word, origin='upper')
    plt.show()

def main():
    # Test of flatten and unflatten. If the size is wrong (e.g. 64
    # when x_size is 128), then it won't graph correctly.
    words, x_size = rasterize_font.flatten_words(rasterize_font.load_words_as_tensor(n = 2000))
    words = rasterize_font.unflatten_words(words, x_size)
    plot_word(words, 1234)
    # And now let's plot a larger size.
    words = rasterize_font.load_words_as_tensor(n = 2000, size=12)
    plot_word(words, 1234)

if __name__ == "__main__":
    main()
