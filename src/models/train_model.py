import argparse
import itertools
import json
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.features import FeatureExtractor, DocFeature, WordFeature
from src.data.datasets import read_cefrj_wordlist, read_cambridge_readability_dataset, \
    read_a1_passages, read_efcamdat_dataset
from src.data.graph import Graph
from src.data.instance import DatasetSplit, NodeType
from src.models.model import GCN
from src.utils import CEFR_LEVELS, cefr_to_beta, accuracy, correlation


def masked_cross_entropy(logit: torch.Tensor,
                         labels: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    # a masked version of cross_entropy where zeroed out elements are ignored
    loss = F.cross_entropy(logit, labels, reduction='none')
    loss = (loss * mask).sum() / mask.sum()

    return loss


def masked_mean_squared_error(logit: torch.Tensor,
                              labels: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    loss = logit.squeeze() - labels
    loss = ((loss * mask) ** 2).sum() / mask.sum()

    return loss


def main():
    # TODO: Parse hyper-parameters from a json config file?
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Enable CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhidden', type=int, nargs='*', default=[16],
                        help='Number of hidden units for each layer.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Mixing weight between word and document losses.')
    parser.add_argument('--word-features', type=str, nargs='*', default=[],
                        help='List of word features to use. If empty, uses identity matrix.')
    parser.add_argument('--doc-features', type=str, nargs='*', default=[],
                        help='List of doc features to use. If empty, uses identity matrix.')
    parser.add_argument('--activation', type=str, default='none',
                        choices=['none', 'relu', 'tanh'],
                        help='Add the specified activation function for each GCN layer.')
    parser.add_argument('--efcamdat-file-path', type=str, default=None,
                        help='Path to EFCamDat. '
                             'If not specified, the dataset will not be used.')
    parser.add_argument('--heads', type=str, default='twin',
                        choices=['single', 'twin'],
                        help='Use either single/same or different linear layer for both word and doc as a final layer.')
    parser.add_argument('--tfidf', action='store_true',
                        help='If specified, weight the adjacency matrix by tf.idf')
    parser.add_argument('--pmi-window-width', type=int, default=-1,
                        help='Window size for calculating PMI, which is disabled when -1')
    parser.add_argument('--conversion', type=str, default='max',
                        choices=['max', 'weighted_sum'],
                        help='If using correlation during evaluation, select whether to convert'
                             'classification to a real value by weighted sum or taking the max.')
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Use either classification or regression loss during training.')
    parser.add_argument('--training-portion', type=int, default=10,
                        help='Specify the amount of training data between 1 (10%) and 10 (100%).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    words = list(read_cefrj_wordlist(training_portion=args.training_portion))
    datasets = [
        read_cambridge_readability_dataset(training_portion=args.training_portion),
        read_a1_passages(training_portion=args.training_portion)]
    if args.efcamdat_file_path:
        datasets.append(read_efcamdat_dataset(args.efcamdat_file_path))
    docs = list(itertools.chain(*datasets))

    # Initialize FeatureExtractor
    doc_value2feat = {feat.value: feat for feat in DocFeature}
    doc_features = {doc_value2feat[value] for value in args.doc_features}
    word_value2feat = {feat.value: feat for feat in WordFeature}
    word_features = {word_value2feat[value] for value in args.word_features}
    feature_extractor = FeatureExtractor(word_features=word_features, doc_features=doc_features,
                                         cuda=args.cuda)

    graph = Graph(feature_extractor)
    graph.add_words(words)
    graph.add_documents(docs)
    num_labeled_docs = sum(1 for doc in docs if doc.label)
    max_word_freq = int(.1 * num_labeled_docs)     # it's too much if a word appears more than 10% of labeled docs
    graph.build_mapping(min_word_freq=3, max_word_freq=max_word_freq, min_document_len=5)
    graph.index()

    print(graph, file=sys.stderr)    # show graph stats

    adj = graph.get_adj(use_tfidf=args.tfidf,
                        pmi_window_width=args.pmi_window_width)

    x = graph.get_feature_matrix()

    type_masks, split_masks = graph.get_type_and_split_masks()
    labels = graph.get_labels()
    labels_beta = torch.Tensor([cefr_to_beta(CEFR_LEVELS[i.item()]) for i in labels])

    # TODO: complete the training pipeline
    # Training
    nclass = 1 if args.mode == "regression" else len(CEFR_LEVELS)

    model = GCN(nfeat=x.shape[1],
                nhidden=args.nhidden,
                nclass=nclass,
                dropout=args.dropout,
                activation=args.activation,
                heads=args.heads)
    if args.cuda:
        adj = adj.cuda()
        model = model.cuda()
        x = x.cuda()
        labels = labels.cuda()
        labels_beta = labels_beta.cuda()
        for k, v in type_masks.items():
            type_masks[k] = v.cuda()
        for k, v in split_masks.items():
            split_masks[k] = v.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    stats = defaultdict(list)
    if not args.efcamdat_file_path:
        stats['num_unlabeled_docs'] = 0
    else:
        stats['num_unlabeled_docs'] = len(list(read_efcamdat_dataset(args.efcamdat_file_path)))
    stats['num_docs'] = graph.get_num_indexed_docs()

    for epoch in range(args.epochs):
        print('Epoch: {:04d}'.format(epoch + 1), file=sys.stderr)

        model.train()
        optimizer.zero_grad()
        logit1, logit2 = model(adj, x)
        if args.mode == "regression":
            loss1 = masked_mean_squared_error(
                logit1, labels_beta, type_masks[NodeType.WORD] * split_masks[DatasetSplit.TRAIN])
            loss2 = masked_mean_squared_error(
                logit2, labels_beta, type_masks[NodeType.DOC] * split_masks[DatasetSplit.TRAIN])
        else:
            loss1 = masked_cross_entropy(
                logit1, labels, type_masks[NodeType.WORD] * split_masks[DatasetSplit.TRAIN])
            loss2 = masked_cross_entropy(
                logit2, labels, type_masks[NodeType.DOC] * split_masks[DatasetSplit.TRAIN])

        loss = (args.alpha * loss1 + (1. - args.alpha) * loss2) * 2
        loss.backward()
        optimizer.step()

        # compute and save loss
        with torch.no_grad():
            if args.mode == "regression":
                dev_loss1 = masked_mean_squared_error(
                    logit1, labels_beta, type_masks[NodeType.WORD] * split_masks[DatasetSplit.DEV])
                dev_loss2 = masked_mean_squared_error(
                    logit2, labels_beta, type_masks[NodeType.DOC] * split_masks[DatasetSplit.DEV])
            else:
                dev_loss1 = masked_cross_entropy(
                    logit1, labels, type_masks[NodeType.WORD] * split_masks[DatasetSplit.DEV])
                dev_loss2 = masked_cross_entropy(
                    logit2, labels, type_masks[NodeType.DOC] * split_masks[DatasetSplit.DEV])
            dev_loss = dev_loss1 + dev_loss2

        print('\tloss: {:.4f}, dev_loss: {:.4f}'.format(loss.item(), dev_loss.item()),
              file=sys.stderr)
        stats['train_loss'].append(loss.item())
        stats['dev_loss'].append(dev_loss.item())

        # compute and save accuracy for train and dev
        model.eval()
        for split in [DatasetSplit.TRAIN, DatasetSplit.DEV]:
            for node_type in [NodeType.WORD, NodeType.DOC]:
                if node_type == NodeType.WORD:
                    logit = logit1
                else:
                    logit = logit2
                acc = accuracy(logit, labels, type_masks[node_type] * split_masks[split],
                               mode=args.mode)

                corr = correlation(logit, labels, type_masks[node_type] * split_masks[split],
                                   mode=args.mode, conversion=args.conversion)

                stats_acc_key = '{}_acc_{}'.format(split.value, node_type.value)
                print('\t{}: {:.4f}'.format(stats_acc_key, acc), file=sys.stderr)
                stats[stats_acc_key].append(acc)

                stats_corr_key = '{}_corr_{}'.format(split.value, node_type.value)
                print('\t{}: {:.4f}'.format(stats_corr_key, corr), file=sys.stderr)
                stats[stats_corr_key].append(corr)

        macro_avg_dev_acc = (stats['dev_acc_word'][-1] + stats['dev_acc_doc'][-1]) / 2
        stats['dev_acc_avr'].append(macro_avg_dev_acc)
        macro_avg_dev_corr = (stats['dev_corr_word'][-1] + stats['dev_corr_doc'][-1]) / 2
        stats['dev_corr_avr'].append(macro_avg_dev_corr)

    # Evaluation
    model.eval()  # turn off dropout (if we are using one)
    logit1, logit2 = model(adj, x)

    print('Evaluation', file=sys.stderr)
    for split in [DatasetSplit.DEV, DatasetSplit.TEST]:
        for node_type in NodeType:
            if node_type == NodeType.WORD:
                logit = logit1
            else:
                logit = logit2

            acc = accuracy(logit, labels, type_masks[node_type] * split_masks[split],
                           mode=args.mode)

            corr = correlation(logit, labels, type_masks[node_type] * split_masks[split],
                               mode=args.mode, conversion=args.conversion)

            stats_key_acc = 'eval_{}_acc_{}'.format(split.value, node_type.value)
            print('\t{}: {:.4f}'.format(stats_key_acc, acc), file=sys.stderr)
            stats[stats_key_acc].append(acc)

            stats_key_corr = 'eval_{}_corr_{}'.format(split.value, node_type.value)
            print('\t{}: {:.4f}'.format(stats_key_corr, corr), file=sys.stderr)
            stats[stats_key_corr].append(corr)

        macro_avg_acc = (stats[f"eval_{split.value}_acc_word"][-1] +
                         stats[f"eval_{split.value}_acc_doc"][-1]) / 2
        print('\teval_{}_acc_avr: {:.4f}'.format(split.value, macro_avg_acc), file=sys.stderr)
        stats[f'eval_{split.value}_acc_avr'].append(macro_avg_acc)

        macro_avg_corr = (stats[f"eval_{split.value}_corr_word"][-1] +
                          stats[f"eval_{split.value}_corr_doc"][-1]) / 2
        print('\teval_{}_corr_avr: {:.4f}'.format(split.value, macro_avg_corr), file=sys.stderr)
        stats[f'eval_{split.value}_corr_avr'].append(macro_avg_corr)

    # Dump stats
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
