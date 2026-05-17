import matplotlib.pyplot as plt
import rasterize_font

def main():
    plt.style.use('_mpl-gallery-nogrid')
    word = rasterize_font.load_words_as_tensor(n = 2000)[1234]
    fig, ax = plt.subplots()
    ax.imshow(word, origin='upper')
    plt.show()

if __name__ == "__main__":
    main()
