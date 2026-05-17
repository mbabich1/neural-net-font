import matplotlib.pyplot as plt
import rasterize_font

def plot_word(words, word_id):
    plt.style.use('_mpl-gallery-nogrid')
    word = words[word_id]
    fig, ax = plt.subplots()
    ax.imshow(word, origin='upper')
    plt.show()

def main():
    plot_word(rasterize_font.load_words_as_tensor(n = 2000), 1234)

if __name__ == "__main__":
    main()
