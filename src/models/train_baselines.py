import argparse
import itertools
import sys
from collections import defaultdict
from typing import List, Union, Dict

import numpy as np
from scipy.stats import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.data.datasets import read_cefrj_wordlist, read_cambridge_readability_dataset, \
    read_a1_passages
from src.data.graph import Graph
from src.data.instance import DatasetSplit
from src.features import FeatureExtractor, DocFeature, WordFeature
from src.utils import cefr_to_beta, beta_to_cefr


def calc_accuracy(preds: List[Union[str, float]],
                  golds: List[Union[str, float]],
                  mode: str) -> float:
    if mode == 'regression':
        preds = [beta_to_cefr(pred) for pred in preds]
    accuracy = sum(1 for pred, gold in zip(preds, golds) if pred == gold)
    accuracy /= len(preds)

    return accuracy


def calc_correlation(preds: List[float], golds: List[float]) -> float:
    rho, _ = stats.spearmanr(preds, golds)

    return rho


def predict_continuous(model: LogisticRegression,
                       xs: List[List[float]],
                       conversion: str) -> List[float]:
    if conversion == 'max':
        preds = model.predict(xs)
        preds = [cefr_to_beta(pred) for pred in preds]
    else:
        # weighted sum
        betas = [cefr_to_beta(c) for c in model.classes_]
        probs = model.predict_proba(xs)
        preds = [sum(p * beta for p, beta in zip(dist, betas)) for dist in probs]

    return preds


def run_pipeline(xs: Dict[DatasetSplit, List[List[float]]],
                 ys_disc: Dict[DatasetSplit, List[str]],
                 ys_cont: Dict[DatasetSplit, List[float]],
                 args: argparse.Namespace) -> None:
    """Run the training using the supplied datasets and show the results."""
    if args.mode == 'classification':
        model = LogisticRegression(multi_class='multinomial')
        model.fit(xs[DatasetSplit.TRAIN], ys_disc[DatasetSplit.TRAIN])
    elif args.mode == 'regression':
        model = LinearRegression()
        model.fit(xs[DatasetSplit.TRAIN], ys_cont[DatasetSplit.TRAIN])
    elif args.mode == 'gradient-boosting-classifier':
        model = GradientBoostingClassifier()
        model.fit(xs[DatasetSplit.TRAIN], ys_disc[DatasetSplit.TRAIN])
    else:
        assert False, 'Invalid model'

    for split in [DatasetSplit.DEV, DatasetSplit.TEST]:
        preds = model.predict(xs[split])
        acc = calc_accuracy(preds, ys_disc[split], mode=args.mode)
        if args.mode == 'classification':
            preds = predict_continuous(model, xs[split], conversion=args.conversion)
        elif args.mode == 'gradient-boosting-classifier':
            assert args.conversion == 'max'
            preds = predict_continuous(model, xs[split], conversion='max')
        corr = calc_correlation(preds, ys_cont[split])
        print('{} acc.: {:.4f}, corr.: {:.4f}'.format(split.value, acc, corr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-features', type=str, nargs='*', default=[],
                        help='List of word features to use. If empty, uses identity matrix.')
    parser.add_argument('--doc-features', type=str, nargs='*', default=[],
                        help='List of doc features to use. If empty, uses identity matrix.')
    parser.add_argument('--conversion', type=str, default='max',
                        choices=['max', 'weighted_sum'],
                        help='If using correlation during evaluation, select whether to convert'
                             'classification to a real value by weighted sum or taking the max.')
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'regression', 'gradient-boosting-classifier'],
                        help='Use either classification or regression loss during training.')
    parser.add_argument('--training-portion', type=int, default=10,
                        help='Specify the amount of training data between 1 (10%) and 10 (100%).')
    args = parser.parse_args()

    words = list(read_cefrj_wordlist(training_portion=args.training_portion))
    docs = list(itertools.chain(
        read_cambridge_readability_dataset(training_portion=args.training_portion),
        read_a1_passages(training_portion=args.training_portion)))

    # Initialize FeatureExtractor
    # TODO: make this a class method of enums
    doc_value2feat = {feat.value: feat for feat in DocFeature}
    doc_features = {doc_value2feat[value] for value in args.doc_features}
    word_value2feat = {feat.value: feat for feat in WordFeature}
    word_features = {word_value2feat[value] for value in args.word_features}
    feature_extractor = FeatureExtractor(word_features=word_features, doc_features=doc_features)

    graph = Graph(feature_extractor=feature_extractor)
    graph.add_words(words)
    graph.add_documents(docs)
    max_word_freq = int(.1 * len(docs))     # it's too much if a word appears more than 10% of docs
    graph.build_mapping(min_word_freq=3, max_word_freq=max_word_freq, min_document_len=5)
    graph.index()

    print(graph, file=sys.stderr)    # show graph stats

    print('words: ')
    xs = defaultdict(list)
    ys_disc = defaultdict(list)     # discrete labels (A1, A2, ..., C2)
    ys_cont = defaultdict(list)     # continuous labels (i.e., betas)
    for word in graph.iterate_indexed_words():
        if word.label is None:
            continue
        feat_vec = []
        for key, value in feature_extractor.word2features(word).items():
            if key == WordFeature.GLOVE:
                value = value.numpy()
                feat_vec.extend(value)
            else:
                feat_vec.append(value)
        xs[word.split].append(feat_vec)
        ys_disc[word.split].append(word.label)
        ys_cont[word.split].append(cefr_to_beta(word.label))
    run_pipeline(xs, ys_disc, ys_cont, args)

    print('docs: ')
    xs = defaultdict(list)
    ys_disc = defaultdict(list)     # discrete labels (A1, A2, ..., C2)
    ys_cont = defaultdict(list)     # continuous labels (i.e., betas)
    for i, doc in enumerate(graph.iterate_indexed_docs()):
        if doc.label is None:
            continue
        feat_vec = []
        for key, value in feature_extractor.doc2features(doc).items():
            if key == DocFeature.BERT_AVG:
                value = value.numpy()
                feat_vec.extend(value)
            else:
                feat_vec.append(value)
        xs[doc.split].append(feat_vec)
        ys_disc[doc.split].append(doc.label)
        ys_cont[doc.split].append(cefr_to_beta(doc.label))
    run_pipeline(xs, ys_disc, ys_cont, args)


if __name__ == '__main__':
    main()
