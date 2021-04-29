import json
import matplotlib.pyplot as plt
import sys
from src.data.instance import DatasetSplit, NodeType


best_macro_avg_stats_key = 'best_dev_acc_avr'

best_dev_stats = {
    NodeType.WORD.value: [],
    NodeType.DOC.value: [],
    'avr': []
}
xs = []

for line in sys.stdin:
    stats = json.loads(line)
    for node_type in [NodeType.WORD, NodeType.DOC]:
        best_stats_key = 'best_{}_acc_{}'.format(DatasetSplit.DEV.value, node_type.value)
        best_dev_stats[node_type.value].append(stats[best_stats_key])
    best_dev_stats["avr"].append(stats[best_macro_avg_stats_key])
    xs.append(stats["num_docs"])

for node_type in [NodeType.WORD, NodeType.DOC]:
    plt.plot(xs, best_dev_stats[node_type.value], label="Best Dev. Accuracy ({})".format(
        node_type.value), marker='o')
plt.plot(xs, best_dev_stats["avr"], label="Best Dev. Accuracy (macro average)", marker='o')
plt.xlabel("Num. of Docs (log scale)")
plt.ylabel("Accuracy")
plt.xscale("symlog")
plt.legend()
plt.savefig("reports/figures/unlabeled_learning_curve.png")
plt.show()