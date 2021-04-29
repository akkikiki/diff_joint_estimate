import matplotlib.pyplot as plt
import numpy as np
from src.utils import read_cefrj_wordlist, word2features, get_log_word_freq
from src.utils import CEFR_LEVELS
plt.rcParams["figure.figsize"] = (15, 7)

CEFR_LEVELS = CEFR_LEVELS[:4]  # C1 and C2 are missing in the CEFR-J dataset
cefr2words = read_cefrj_wordlist()


def visualize_log_word_freq():
    fig, ax = plt.subplots()
    mean_freq_levels = []
    median_freq_levels = []

    word_freqs = get_log_word_freq()

    for i, cefr_level in enumerate(CEFR_LEVELS):
        word_freq = np.array([word2features(word, word_freq=word_freqs)['freq'] for word in cefr2words[cefr_level]])
        mean_freq_levels.append(np.mean(word_freq))
        median_freq_levels.append(np.median(word_freq))

    y_pos = np.arange(len(mean_freq_levels))
    width = 0.35  # the width of the bars
    ax.bar(y_pos - width / 2, mean_freq_levels, width, label='mean frequency')
    ax.bar(y_pos + width / 2, median_freq_levels, width, label='median frequency')

    plt.xticks(y_pos, CEFR_LEVELS)

    plt.title("Word Frequency on each CEFR Level", fontsize=14)
    plt.xlabel("CEFR Levels", fontsize=14)
    plt.ylabel("Negative log Word Frequency P(W) in EN Wikipedia", fontsize=14)
    plt.legend()
    out_filename = "reports/figures/word_cefr/word_freq_wikipedia.png"
    plt.savefig(out_filename)
    print(f"Plot created at {out_filename}")


def visualize_lengths():
    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i, cefr_level in enumerate(CEFR_LEVELS):
        word_length = np.array([word2features(word)['length'] for word in cefr2words[cefr_level]])
        ax = axes[i]
        ax.hist(word_length, bins=10)
        ax.set_title(f"CEFR: {cefr_level}", fontsize=14)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 1200)
        ax.grid(True)
    axes[0].set_ylabel("Occurrences", fontsize=14)
    fig.text(0.5, 0.04, "Length of Words (Num. of Characters)", ha="center", va="center", fontsize=14)
    out_filename = "reports/figures/word_cefr/word_len.png"
    plt.savefig(out_filename)
    print(f"Plot created at {out_filename}")

if __name__ == "__main__":
    visualize_lengths()
    visualize_log_word_freq()
