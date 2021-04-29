import argparse
import itertools
import json
import sys

import numpy as np
import torch

from src.data.datasets import read_cefrj_wordlist, read_cambridge_readability_dataset, \
    read_a1_passages
from src.data.graph import Graph
from src.data.instance import DatasetSplit, NodeType
from src.utils import CEFR2INT, beta_to_cefr, cefr_to_beta, accuracy, correlation


def get_preds(beta_docs: np.ndarray,
              beta_words: np.ndarray,
              graph: Graph) -> torch.Tensor:
    """Given betas for docs and words, return a pred tensor that matches GCN's ouput"""
    preds = torch.zeros((graph.get_num_nodes(), 1))
    for doc in graph.iterate_indexed_docs():
        preds[doc.doc_id + graph.get_node_offset(NodeType.DOC)] = beta_docs[doc.doc_id]

    for word in graph.iterate_indexed_words():
        preds[word.word_id + graph.get_node_offset(NodeType.WORD)] = beta_words[word.word_id]

    return preds


def main():
    np.random.seed(31415926)

    parser = argparse.ArgumentParser()
    parser.add_argument('out_path_docs', help='Output path for document data')
    parser.add_argument('out_path_words', help='Output path for word data')
    parser.add_argument('--learning_rate', help='Learning rate', type=float, default=.1)
    parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=30)
    parser.add_argument('--supervised', help='If specified, fix beta for labeled nodes',
                        action='store_true')
    args = parser.parse_args()

    words = list(read_cefrj_wordlist())
    docs = list(itertools.chain(read_cambridge_readability_dataset(),
                                read_a1_passages()))

    graph = Graph(feature_extractor=None)
    graph.add_words(words)
    graph.add_documents(docs)
    max_word_freq = int(.1 * len(docs))     # it's too much if a word appears more than 10% of docs
    graph.build_mapping(min_word_freq=3, max_word_freq=max_word_freq, min_document_len=5)
    graph.index()

    print(graph)    # show graph stats

    labels = graph.get_labels()
    type_masks, split_masks = graph.get_type_and_split_masks()

    # initialize difficulty parameter beta using a gaussian N(0, 1)
    beta_docs = np.random.normal(loc=0., scale=1.,
                                 size=graph.get_num_indexed_docs())
    beta_words = np.random.normal(loc=0., scale=1.,
                                  size=graph.get_num_indexed_words())
    if args.supervised:
        for doc in graph.iterate_indexed_docs():
            if doc.split == DatasetSplit.TRAIN:
                beta_docs[doc.doc_id] = cefr_to_beta(doc.label)

        for word in graph.iterate_indexed_words():
            if word.split == DatasetSplit.TRAIN and word.label:
                beta_words[word.word_id] = cefr_to_beta(word.label)

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    print(f'learning_rate = {learning_rate}, num_epochs = {num_epochs}', file=sys.stderr)

    for epoch in range(num_epochs):
        print(f'epoch = {epoch}', file=sys.stderr)
        for doc in graph.iterate_indexed_docs():
            if args.supervised and doc.split == DatasetSplit.TRAIN:
                continue
            max_beta = max(beta_words[w_id] for w_id in doc.word_ids)
            grad = max_beta - beta_docs[doc.doc_id]
            beta_docs[doc.doc_id] += learning_rate * grad

        for word in graph.iterate_indexed_words():
            if not word.doc_ids:
                continue
            if args.supervised and word.split == DatasetSplit.TRAIN:
                continue
            min_beta = min(beta_docs[d_id] for d_id in word.doc_ids)
            grad = min_beta - beta_words[word.word_id]
            beta_words[word.word_id] += learning_rate * grad

        preds = get_preds(beta_docs, beta_words, graph)
        for split in [DatasetSplit.TRAIN, DatasetSplit.DEV]:
            word_mask = type_masks[NodeType.WORD] * split_masks[split]
            doc_mask = type_masks[NodeType.DOC] * split_masks[split]

            word_acc = accuracy(preds, labels, word_mask, mode='regression')
            doc_acc = accuracy(preds, labels, doc_mask, mode='regression')
            avr_acc = (word_acc + doc_acc) / 2
            print('{} acc: word: {:.4f}, doc: {:.4f}, macro avr: {:.4f}'.format(
                split.value, word_acc, doc_acc, avr_acc))

            word_corr = correlation(preds, labels, word_mask, mode='regression')
            doc_corr = correlation(preds, labels, doc_mask, mode='regression')
            avr_corr = (word_corr + doc_corr) / 2
            print('{} corr: word: {:.4f}, doc: {:.4f}, macro avr: {:.4f}'.format(
                split.value, word_corr, doc_corr, avr_corr))

    word_mask = type_masks[NodeType.WORD] * split_masks[DatasetSplit.TEST]
    doc_mask = type_masks[NodeType.DOC] * split_masks[DatasetSplit.TEST]

    test_word_acc = accuracy(preds, labels, word_mask, mode='regression')
    test_doc_acc = accuracy(preds, labels, doc_mask, mode='regression')
    test_avr_acc = (test_word_acc + test_doc_acc) / 2
    print('test acc: word: {:.4f}, doc: {:.4f}, macro avr: {:.4f}'.format(
        test_word_acc, test_doc_acc, test_avr_acc))

    test_word_corr = correlation(preds, labels, word_mask, mode='regression')
    test_doc_corr = correlation(preds, labels, doc_mask, mode='regression')
    test_avr_corr = (test_word_corr + test_doc_corr) / 2
    print('test corr: word: {:.4f}, doc: {:.4f}, macro avr: {:.4f}'.format(
        test_word_corr, test_doc_corr, test_avr_corr))

    with open(args.out_path_docs, mode='w') as f:
        for doc in graph.iterate_indexed_docs():
            if doc.split != DatasetSplit.TEST:
                continue
            data = {
                'id': doc.doc_id,
                'doc': doc.doc_str,
                'label': doc.label,
                'words': doc.words,
                'beta': beta_docs[doc.doc_id]
            }
            print(json.dumps(data), file=f)

    with open(args.out_path_words, mode='w') as f:
        for word in graph.iterate_indexed_words():
            if word.split != DatasetSplit.TEST:
                continue
            data = {
                'id': word.word_id,
                'word': word.word_str,
                'label': word.label,
                'docs': word.docs,
                'beta': beta_words[word.word_id]
            }
            print(json.dumps(data), file=f)


if __name__ == '__main__':
    main()
