import matplotlib.pyplot as plt
from src.utils import read_cambridge_readability_dataset, passage2features
from src.utils import TEST2CEFR
plt.rcParams["figure.figsize"] = (15, 7)

exam2passages = read_cambridge_readability_dataset()
max_tokens = max(len(passage) for passages in exam2passages.values() for passage in passages)

fig, axes = plt.subplots(nrows=1, ncols=5)

for i, (exam, passages) in enumerate(sorted(exam2passages.items(), key=lambda x: TEST2CEFR[x[0]])):
    title = f"{exam} ({TEST2CEFR[exam]})"
    sentence_lengths = [passage2features(passage)['length'] for passage in passages]
    ax = axes[i]
    ax.set_title(title)
    ax.hist(sentence_lengths, bins=10)
    ax.set_xlim(0, max_tokens)
    ax.set_ylim(0, 1200)
    ax.grid(True)

axes[0].set_ylabel("Occurrences", fontsize=14)
fig.text(0.5, 0.04, "Num. of tokens per sentence", ha="center", va="center", fontsize=14)

out_filename = "reports/figures/sentence_cefr/sentence_len.png"
plt.savefig(out_filename)
print(f"Plot created at {out_filename}")
