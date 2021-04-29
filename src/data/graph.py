import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import scipy as sp
import torch

from src.data.instance import Word, Document, DatasetSplit, NodeType
from src.utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, iter_ngrams
from src.utils import CEFR2INT
from src.features import FeatureExtractor, WordFeature, DocFeature


class Graph:
    """Manages the graph data structure as well as the mapping from word/documents to IDs"""

    def __init__(self, feature_extractor: Optional[FeatureExtractor]):
        self.words: Dict[str, Word] = {}
        self.docs: Dict[str, Document] = {}
        self.word2id: Optional[Dict[str, int]] = None
        self.doc2id: Optional[Dict[str, int]] = None
        self.id2word: Optional[Dict[int, Word]] = None
        self.id2doc: Optional[Dict[int, Document]] = None
        self.feature_extractor = feature_extractor

    def add_words(self, words: Iterable[Word]):
        for word in words:
            if word.word_str in self.words:
                continue
            self.words[word.word_str] = word

    def add_documents(self, documents: Iterable[Document]):
        for doc in documents:
            self.docs[doc.doc_str] = doc

            for word_str in doc.words:
                word = self.words.get(word_str)
                if word is None:
                    word = Word(word_str=word_str,
                                word_id=None,
                                label=None,
                                split=DatasetSplit.UNLABELED,
                                docs=[],
                                doc_ids=None)
                    self.words[word_str] = word

                # TODO: should we unique words?
                word.docs.append(doc.doc_str)

    def get_feature_matrix(self) -> torch.Tensor:
        """If no features are specified in FeatureExtractor, then use the identity matrix."""
        if self.feature_extractor.doc_features:
            x_docs = []
            for doc in sorted(self.iterate_indexed_docs(), key=lambda doc: doc.doc_id):
                # TODO: support more features
                # TODO: use DictVectorizer
                features = self.feature_extractor.doc2features(doc)

                x_doc = []
                for feature_name, feature_value in features.items():
                    if feature_name == DocFeature.BERT_CLS or feature_name == DocFeature.BERT_AVG:
                        x_doc.append(feature_value)
                    else:
                        feature = torch.FloatTensor([feature_value])
                        if self.feature_extractor.cuda:
                            feature = feature.cuda()
                        x_doc.append(feature)

                x_docs.append(torch.cat(x_doc))
            x_docs = torch.stack(x_docs, dim=0)
        else:
            x_docs = torch.eye(self.get_num_indexed_docs())

        if self.feature_extractor.word_features:
            x_words = []
            for word in sorted(self.iterate_indexed_words(), key=lambda word: word.word_id):
                features = self.feature_extractor.word2features(word)
                # TODO: support more features

                x_word = []
                for feature_name, feature_value in features.items():
                    if feature_name == WordFeature.GLOVE:
                        x_word.append(feature_value)
                    else:
                        x_word.append(torch.FloatTensor([feature_value]))

                x_words.append(torch.cat(x_word))
            x_words = torch.stack(x_words, dim=0)
        else:
            x_words = torch.eye(self.get_num_indexed_words())

        # Handle word and document padding
        x_words_padding = torch.zeros(self.get_num_indexed_words(), x_docs.shape[1])
        x_docs_padding = torch.zeros(self.get_num_indexed_docs(), x_words.shape[1])
        if self.feature_extractor.cuda:
            x_words_padding = x_words_padding.cuda()
            x_docs_padding = x_docs_padding.cuda()
            x_words = x_words.cuda()
            x_docs = x_docs.cuda()

        x_words_padded = torch.cat((x_words, x_words_padding), dim=1)
        x_docs_padded = torch.cat((x_docs_padding, x_docs), dim=1)

        # Confirm that words and docs end up with the same number of features
        assert x_words_padded.shape[1] == x_docs_padded.shape[1]

        return torch.cat((x_words_padded, x_docs_padded), dim=0)

    def build_mapping(self,
                      min_word_freq: int = 0,
                      max_word_freq: int = 0,
                      min_document_len: int = 0):
        doc_id = 0
        self.doc2id = {}
        self.id2doc = {}
        for doc_str, doc in self.docs.items():
            if len(doc.words) < min_document_len:
                continue

            self.doc2id[doc_str] = doc_id
            self.id2doc[doc_id] = doc
            doc_id += 1

        word_id = 1
        oov = Word('OOV', label=None, docs=[], split=DatasetSplit.UNLABELED)
        self.word2id = {'OOV': 0}   # pre-assign OOV token
        self.id2word = {0: oov}
        for word_str, word in self.words.items():
            if len(word.docs) < min_word_freq:
                continue

            if 0 < max_word_freq < len(word.docs):
                continue

            self.word2id[word_str] = word_id
            self.id2word[word_id] = word
            word_id += 1

        self.words['OOV'] = oov

    def index(self):
        for doc in self.docs.values():
            doc.doc_id = self.doc2id.get(doc.doc_str)
            doc.word_ids = []
            for word_str in doc.words:
                if word_str in self.word2id:
                    doc.word_ids.append(self.word2id[word_str])
                else:
                    doc.word_ids.append(self.word2id['OOV'])

        for word in self.words.values():
            word.word_id = self.word2id.get(word.word_str)
            word.doc_ids = []
            for doc_str in word.docs:
                if doc_str in self.doc2id:
                    word.doc_ids.append(self.doc2id[doc_str])

    def get_doc(self, doc_id: int) -> Optional[Document]:
        return self.id2doc.get(doc_id)

    def get_word(self, word_id: int) -> Optional[Word]:
        return self.id2word.get(word_id)

    def get_num_indexed_docs(self) -> int:
        return len(self.doc2id)

    def get_num_indexed_words(self) -> int:
        return len(self.word2id)

    def get_num_nodes(self) -> int:
        return self.get_num_indexed_words() + self.get_num_indexed_docs()

    def iterate_indexed_docs(self) -> Iterable[Document]:
        for doc in self.docs.values():
            if doc.doc_id is not None:
                yield doc

    def iterate_indexed_words(self) -> Iterable[Word]:
        for word in self.words.values():
            if word.word_id is not None:
                yield word

    def get_node_offset(self, node_type: NodeType) -> int:
        """Given a node type, returns an offset for the (global) node indices."""
        offsets = {
            NodeType.WORD: 0,
            NodeType.DOC: self.get_num_indexed_words()
        }

        return offsets[node_type]

    def _calculate_pmi(self, window_width: int) -> Tuple[Dict[int, Dict[int, float]], float]:
        word_counts = Counter()
        cooccur_counts = defaultdict(Counter)
        pmi = defaultdict(Counter)
        max_pmi = 0.

        for doc in self.iterate_indexed_docs():
            for ngram in iter_ngrams(doc.word_ids, n=window_width):
                for i, src_word_id in enumerate(ngram):
                    if src_word_id == 0:    # OOV
                        continue
                    word_counts[src_word_id] += 1
                    for j, tgt_word_id in enumerate(ngram):
                        if i != j and tgt_word_id != 0:
                            cooccur_counts[src_word_id][tgt_word_id] += 1

        log_total_counts = math.log(sum(word_counts.values()))
        for src_word_id, tgt_word_counts in cooccur_counts.items():
            for tgt_word_id, counts in tgt_word_counts.items():
                unconstrained_pmi = log_total_counts + math.log(counts)
                unconstrained_pmi -= math.log(word_counts[src_word_id] *
                                              word_counts[tgt_word_id])
                if unconstrained_pmi > 0.:
                    pmi[src_word_id][tgt_word_id] = unconstrained_pmi
                    max_pmi = max(max_pmi, unconstrained_pmi)

        return pmi, max_pmi

    def get_adj(self,
                use_tfidf: bool = False,
                pmi_window_width: int = -1) -> torch.sparse.FloatTensor:

        num_nodes = self.get_num_nodes()
        adj = sp.sparse.dok_matrix((num_nodes, num_nodes), dtype=np.float32)

        word_offset = self.get_node_offset(NodeType.WORD)
        doc_offset = self.get_node_offset(NodeType.DOC)

        max_idf = math.log(self.get_num_indexed_docs())
        for doc in self.iterate_indexed_docs():
            for word_id, tf in Counter(doc.word_ids).items():
                word = self.get_word(word_id)
                if word.word_str == 'OOV':
                    continue

                if use_tfidf:
                    # compute normalized idf
                    # idf(w)/max_w idf(w) = log(N / df(w)) / log(N)
                    idf = 1. - math.log(len(set(word.doc_ids))) / max_idf
                    weight = tf * idf
                else:
                    weight = 1.

                adj[word.word_id + word_offset, doc.doc_id + doc_offset] = weight
                adj[doc.doc_id + doc_offset, word.word_id + word_offset] = weight

        for i in range(num_nodes):
            adj[i, i] = 1.  # self-loop

        if pmi_window_width > 0:
            pmi, max_pmi = self._calculate_pmi(window_width=pmi_window_width)
            for src_word_id, tgt_word_pmis in pmi.items():
                for tgt_word_id, value in tgt_word_pmis.items():
                    adj[src_word_id + word_offset, tgt_word_id + word_offset] = value / max_pmi

        adj = normalize_adj(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        return adj

    def get_type_and_split_masks(self) -> Tuple[Dict[NodeType, torch.Tensor],
                                                Dict[DatasetSplit, torch.Tensor]]:
        num_words = self.get_num_indexed_words()
        num_docs = self.get_num_indexed_docs()
        num_nodes = self.get_num_nodes()
        type_masks = {
            NodeType.WORD: torch.cat([torch.ones((num_words,), dtype=torch.long),
                                      torch.zeros((num_docs,), dtype=torch.long)]),
            NodeType.DOC: torch.cat([torch.zeros((num_words,), dtype=torch.long),
                                     torch.ones((num_docs,), dtype=torch.long)]),

        }
        split_masks = {
            DatasetSplit.TRAIN: torch.zeros((num_nodes,), dtype=torch.long),
            DatasetSplit.DEV: torch.zeros((num_nodes,), dtype=torch.long),
            DatasetSplit.TEST: torch.zeros((num_nodes,), dtype=torch.long),
        }

        word_offset = self.get_node_offset(NodeType.WORD)
        for word in self.iterate_indexed_words():
            if word.label and word.split != DatasetSplit.UNLABELED:
                split_masks[word.split][word.word_id + word_offset] = 1

        doc_offset = self.get_node_offset(NodeType.DOC)
        for doc in self.iterate_indexed_docs():
            if doc.label and doc.split != DatasetSplit.UNLABELED:
                split_masks[doc.split][doc.doc_id + doc_offset] = 1

        return type_masks, split_masks

    def get_labels(self) -> torch.LongTensor:
        labels = torch.zeros((self.get_num_nodes(),), dtype=torch.long)
        word_offset = self.get_node_offset(NodeType.WORD)
        for word in self.iterate_indexed_words():
            if word.label:
                labels[word.word_id + word_offset] = CEFR2INT[word.label]

        doc_offset = self.get_node_offset(NodeType.DOC)
        for doc in self.iterate_indexed_docs():
            if doc.label:
                labels[doc.doc_id + doc_offset] = CEFR2INT[doc.label]

        return labels

    def __str__(self):
        return ('Graph: \n'
                f'\t# of total docs:   {len(self.docs)}\n'
                f'\t# of total words:  {len(self.words)}\n'
                f'\t# of mapped docs:  {len(self.doc2id)}\n'
                f'\t# of mapped words: {len(self.word2id)}')
