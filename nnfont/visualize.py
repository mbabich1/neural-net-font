import matplotlib.pyplot as plt
import torch

from nnfont.rasterize_font import load_words_as_tensor
from nnfont.rasterize_font import flatten_words, unflatten_words

def plot_word(word, img_id):
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(word, origin='upper')
    plt.savefig(img_id)

def plot_words_grid(words, img_id):
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots(len(words), 1)
    for i in range(len(words)):
        ax[i].imshow(words[i], origin='upper')

    plt.savefig(img_id)
    plt.close(fig)

def plot_multi_words(words, ids, save_id):
    plt.style.use('_mpl-gallery-nogrid')
    word = [words[id] for id in ids]
    fig, ax = plt.subplots(len(ids), 1, layout='constrained')
    for i in range(len(ids)):
        ax[i].imshow(word[i], origin='upper')

    plt.savefig(save_id)
    plt.close(fig)

def show_word_plot():
    plt.show()

def main():
    # Test of flatten and unflatten. If the size is wrong (e.g. 64
    # when x_size is 128), then it won't graph correctly.
    words8, x_size = flatten_words(load_words_as_tensor(size = 8, n = 2000))
    words8 = unflatten_words(words8, x_size)
    # Test of two different sizes.
    words12 = load_words_as_tensor(n = 2000, size = 12)
    a = words8[1234]
    b = words12[1234]
    c = torch.zeros(b.shape, dtype=torch.uint8)
    c[0 : a.shape[0], 0 : a.shape[1]] += a
    words_to_plot = torch.stack((c, b))
    plot_multi_words(words_to_plot, [0, 1])
    show_word_plot()
    # Plot multiple from the same tensor
    plot_multi_words(words12, [1234, 1999, 333, 555])
    show_word_plot()
    # Load different faces
    ttf = ['NotoSans-Regular.ttf',
           'NotoSans-BoldItalic.ttf',
           'NotoSans-Bold.ttf',
           'NotoSans-Italic.ttf',
           'NotoSerif-Regular.ttf',
           'NotoSerif-BoldItalic.ttf',
           'NotoSerif-Bold.ttf',
           'NotoSerif-Italic.ttf']
    all_words = [load_words_as_tensor(n = 200, size = 12, font_path = font)
                 for font in ttf]
    selected_word = [data[50] for data in all_words]
    fonts_word = torch.stack(selected_word)
    plot_multi_words(fonts_word,
                     [i for i in range(0,
                                       len(all_words) // 2)])
    show_word_plot()
    plot_multi_words(fonts_word,
                     [i for i in range(len(all_words) // 2,
                                       len(all_words))])
    show_word_plot()

if __name__ == "__main__":
    main()
