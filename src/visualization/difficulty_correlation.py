import itertools
from collections import Counter

from src.data.datasets import read_cefrj_wordlist, read_cambridge_readability_dataset, \
    read_a1_passages
from src.data.graph import Graph
from src.utils import CEFR_LEVELS


def median(values):
    return values[len(values) // 2]


def main():
    words = list(read_cefrj_wordlist())
    datasets = [read_cambridge_readability_dataset(),
                read_a1_passages()]
    docs = list(itertools.chain(*datasets))

    graph = Graph(feature_extractor=None)
    graph.add_words(words)
    graph.add_documents(docs)
    max_word_freq = int(.1 * len(docs))     # it's too much if a word appears more than 10% of docs
    graph.build_mapping(min_word_freq=3, max_word_freq=max_word_freq, min_document_len=5)
    graph.index()

    funcs = [(min, 'min'), (median, 'median'), (max, 'max')]
    print('difficulty of documents vs func(difficulties of words):')
    for func, func_name in funcs:
        level_pair_count = Counter()
        for doc in graph.iterate_indexed_docs():
            word_levels = []
            for word_id in doc.word_ids:
                word = graph.get_word(word_id)
                if word.label is None:
                    continue
                word_levels.append(word.label)

            if word_levels:
                word_levels.sort()
                level_pair_count[(doc.label, func(word_levels))] += 1

        print(func_name)
        print('\t'.join([''] + CEFR_LEVELS))
        for doc_level in CEFR_LEVELS:
            row_strs = [doc_level]
            row_strs.extend(str(level_pair_count[(doc_level, level)])
                            for level in CEFR_LEVELS)
            print('\t'.join(row_strs))

    print('difficulty of words vs func(difficulties of documents):')
    for func, func_name in funcs:
        level_pair_count = Counter()
        for word in graph.iterate_indexed_words():
            doc_levels = []
            for doc_id in word.doc_ids:
                doc = graph.get_doc(doc_id)
                if doc.label is None:
                    continue
                doc_levels.append(doc.label)

            if doc_levels:
                doc_levels.sort()
                level_pair_count[(word.label, func(doc_levels))] += 1

        print(func_name)
        print('\t'.join([''] + CEFR_LEVELS))
        for word_level in CEFR_LEVELS:
            row_strs = [word_level]
            row_strs.extend(str(level_pair_count[(word_level, level)])
                            for level in CEFR_LEVELS)
            print('\t'.join(row_strs))


if __name__ == '__main__':
    main()
