import json
import sys

import matplotlib.pyplot as plt

stats = json.load(sys.stdin)

plt.plot(stats['train_loss'], label="Training")
plt.plot(stats['dev_loss'], label="Dev")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("reports/figures/loss_per_epoch.png")
plt.show()
