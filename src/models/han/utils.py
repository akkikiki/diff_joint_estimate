"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from typing import List, Tuple

import torch
from src.data.instance import Document
from nltk.tokenize import sent_tokenize, word_tokenize


def matrix_mul(input: torch.Tensor,
               weight: torch.Tensor,
               bias=False) -> torch.Tensor:
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()


def element_wise_mul(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)


def get_max_lengths(docs: List[Document]) -> Tuple[int, int]:
    word_length_list = []
    sent_length_list = []
    for doc in docs:
        text = doc.text
        sent_list = sent_tokenize(text)
        sent_length_list.append(len(sent_list))

        for sent in sent_list:
            word_list = word_tokenize(sent)
            word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]
