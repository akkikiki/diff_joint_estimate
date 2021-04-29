import json
import sys

import matplotlib.pyplot as plt

stats = json.load(sys.stdin)


plt.plot(stats['train_acc_word'], label="Train Word Accuracy")
plt.plot(stats['train_acc_doc'], label="Train Doc Accuracy")
plt.plot(stats['dev_acc_word'], label="Dev. Word Accuracy")
plt.plot(stats['dev_acc_doc'], label="Dev. Doc Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("reports/figures/acc_per_epoch.png")
plt.show()
