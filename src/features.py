import numpy as np
import torch

from collections import Counter
from enum import Enum
from transformers import BertModel, BertTokenizer
from typing import Any, Dict, Set

from src.data.instance import Document, Word


class WordFeature(Enum):
    GLOVE = 'glove'
    LENGTH = 'length'
    FREQ = 'freq'


class DocFeature(Enum):
    BERT_CLS = 'bert_cls'  # Use CLS token as document embedding
    BERT_AVG = 'bert_avg'  # Use average of the WordPiece tokens as document embedding
    LENGTH = 'length'


class FeatureExtractor:
    def __init__(self,
                 word_features: Set[WordFeature],
                 doc_features: Set[DocFeature],
                 cuda: bool = False):
        self.GLOVE_DIM = 300
        self.BERT_DIM = 768
        self.cuda = cuda

        if WordFeature.FREQ in word_features:
            self.word_freq = self.build_log_word_freq()

        if WordFeature.GLOVE in word_features:
            self.word2embed = self.read_glove()

        assert not (DocFeature.BERT_AVG in doc_features and DocFeature.BERT_CLS in doc_features)

        if DocFeature.BERT_CLS in doc_features or DocFeature.BERT_AVG in doc_features:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if self.cuda:
                self.bert_model = self.bert_model.cuda()

        self.word_features = word_features
        self.doc_features = doc_features

    def build_log_word_freq(self) -> Dict[str, float]:
        neg_log_token_freq = Counter()
        total_word_count = 0
        with open("data/processed/token_count_wikipedia_freq10.txt") as f:
            for line in f:
                word, count = line.strip().split()
                total_word_count += int(count)
                neg_log_token_freq[word] = int(count)

        log_total_word_count = np.log(total_word_count)
        for word in neg_log_token_freq.keys():
            word_count = neg_log_token_freq[word]
            neg_log_token_freq[word] = -1 * (np.log(word_count) - log_total_word_count)

        return neg_log_token_freq

    def doc2features(self,
                     doc: Document) -> Dict[DocFeature, Any]:
        """Compute features for a Document object"""
        result = {}

        if DocFeature.BERT_AVG in self.doc_features:
            result[DocFeature.BERT_AVG] = self.get_doc_embedding(doc)

        if DocFeature.BERT_CLS in self.doc_features:
            result[DocFeature.BERT_CLS] = self.get_doc_embedding(doc)

        if DocFeature.LENGTH in self.doc_features:
            result[DocFeature.LENGTH] = len(doc.word_ids)

        return result

    def word2features(self,
                      word: Word) -> Dict[WordFeature, Any]:
        """Compute features for a Word object"""
        result = {}

        if WordFeature.GLOVE in self.word_features:
            result[WordFeature.GLOVE] = self.get_word_embedding(word)

        if WordFeature.LENGTH in self.word_features:
            result[WordFeature.LENGTH] = len(word.word_str)

        if WordFeature.FREQ in self.word_features:
            result[WordFeature.FREQ] = self.word_freq.get(word.word_str, 0)

        return result

    def get_doc_embedding(self,
                          doc: Document) -> torch.Tensor:
        """Obtain document embeddings by taking the last layer of CLS embedding"""
        encoded_doc = self.tokenizer.encode(doc.text, max_length=512)
        input_ids = torch.tensor(encoded_doc).unsqueeze(0)
        if self.cuda:
            input_ids = input_ids.cuda()
        with torch.no_grad():
            outputs = self.bert_model(input_ids)
        last_hidden_states = outputs[0].squeeze(0)
        if DocFeature.BERT_CLS in self.doc_features:
            doc_embedding = last_hidden_states[0]  # CLS
        elif DocFeature.BERT_AVG in self.doc_features:
            doc_embedding = torch.mean(last_hidden_states, dim=0)  # average of all tokens

        return doc_embedding

    def get_word_embedding(self,
                           word: Word) -> torch.Tensor:
        """Obtain word embeddings using GloVe"""
        if word.word_str in self.word2embed:
            return torch.from_numpy(self.word2embed[word.word_str])
        else:
            return torch.zeros(self.GLOVE_DIM, dtype=torch.float)  # OOV

    def read_glove(self) -> Dict[str, np.ndarray]:
        word2embed = {}
        with open("data/raw/glove/glove.6B.300d.txt") as f:
            for line in f:
                word, vec = line.strip().split(' ', maxsplit=1)
                word2embed[word] = np.fromstring(vec, dtype=np.float32, sep=' ')
        return word2embed
