import matplotlib.pyplot as plt
import rasterize_font

def plot_word(word, img_id):
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(word, origin='upper')
    plt.savefig(img_id)

def plot_multi_words(words, ids):
    plt.style.use('_mpl-gallery-nogrid')
    word = [words[id] for id in ids]
    fig, ax = plt.subplots(len(ids), 1, layout='constrained')
    for i in range(len(ids)):
        ax[i].imshow(word[i], origin='upper')

def show_word_plot():
    plt.show()

def main():
    # Test of flatten and unflatten. If the size is wrong (e.g. 64
    # when x_size is 128), then it won't graph correctly.
    words, x_size = rasterize_font.flatten_words(rasterize_font.load_words_as_tensor(n = 2000))
    words = rasterize_font.unflatten_words(words, x_size)
    plot_word(words, 1234)
    show_word_plot()
    # And now let's plot a larger size.
    words = rasterize_font.load_words_as_tensor(n = 2000, size=12)
    plot_word(words, 1234)
    show_word_plot()
    plot_multi_words(words, [1234, 1999, 333, 555])
    show_word_plot()

if __name__ == "__main__":
    main()
