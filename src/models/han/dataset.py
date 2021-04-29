"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from collections import defaultdict
from typing import List

from torch.utils.data.dataset import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from src.data.instance import DatasetSplit, Document
from src.utils import CEFR2INT


class MyDataset(Dataset):

    def __init__(self,
                 docs: List[Document],
                 split: DatasetSplit,
                 max_length_sentences: int,
                 max_length_word: int):
        super(MyDataset, self).__init__()

        self.dict = defaultdict(int)
        texts, labels = [], []
        tokens = []
        for doc in docs:
            if doc.split == split:
                texts.append(doc.text)
                labels.append(doc.label)
                tokens.append(doc.words)
                for token in doc.words:
                    if not token in self.dict:
                        self.dict[token] = len(self.dict)

        self.texts = texts
        self.labels = labels
        self.tokens = tokens
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,
                    index: int):
        label = CEFR2INT[self.labels[index]]
        text = self.texts[index]
        document_encode = [
            [self.dict[word] if word in self.dict else -1 for word in word_tokenize(
                text=sentence)] for sentence in sent_tokenize(text=text)
        ]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label
