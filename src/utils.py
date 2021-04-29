import re
from typing import List

import numpy as np
import scipy.sparse as sps
import scipy.stats as stats
import torch
import torch.nn.functional as F

CEFR_LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
CEFR2INT = {c: i for i, c in enumerate(CEFR_LEVELS)}

# Build the UK to US word list, taken from
# http://www.tysto.com/uk-us-spelling-list.html
UK2US_WORDS = {}

with open('data/processed/uk_us_spelling_list.txt') as f:
    for line in f:
        uk_word, us_word = line[:-1].split('\t')
        UK2US_WORDS[uk_word] = us_word


def cefr_to_beta(level: str) -> float:
    """Convert discrete CEFR labels to continuous beta which is normally distributed.
    See item response theory (IRT) for why this is called beta.
    https://en.wikipedia.org/wiki/Item_response_theory

    Percent point function (ppf) below is the inverse of the
    cumulative distribution function of N(0, 1) gaussian."""

    return {
        'A1': -1.382994127100638,   # ppf(0.5/6)
        'A2': -0.6744897501960817,  # ppf(1.5/6)
        'B1': -0.2104283942479247,  # ppf(2.5/6)
        'B2': 0.21042839424792484,  # ppf(3.5/6)
        'C1': 0.6744897501960817,   # ppf(4.5/6)
        'C2': 1.382994127100638     # ppf(5.5/6)
    }[level]


def beta_to_cefr(beta: float) -> str:
    """Inverse of cefr_to_beta() above."""
    thresholds = [
        -0.967421566101701,     # ppf(1/6)
        -0.43072729929545756,   # ppf(2/6)
        0.0,                    # ppf(3/6)
        0.43072729929545744,    # ppf(4/6)
        0.967421566101701       # ppf(5/6)
    ]

    if beta < thresholds[0]:
        return 'A1'
    elif thresholds[0] <= beta < thresholds[1]:
        return 'A2'
    elif thresholds[1] <= beta < thresholds[2]:
        return 'B1'
    elif thresholds[2] <= beta < thresholds[3]:
        return 'B2'
    elif thresholds[3] <= beta < thresholds[4]:
        return 'C1'
    elif thresholds[4] <= beta:
        return 'C2'


def normalize_tokens(tokens: List[str]) -> List[str]:
    tokens_normalized = []
    for token in tokens:
        if not re.match(r'[A-Za-z]+', token):
            continue
        token = token.lower()
        token = UK2US_WORDS.get(token, token)

        tokens_normalized.append(token)

    return tokens_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output: torch.Tensor,
             labels: torch.Tensor,
             mask: torch.Tensor,
             mode: str) -> float:
    if mode == "regression":
        preds = mask.new_tensor([CEFR2INT[beta_to_cefr(i.item())] for i in output])
    elif mode == "classification":
        preds = output.argmax(dim=1)
    else:
        raise ValueError('Select mode from regression or classification')
    correct = preds.eq(labels).double()
    correct *= mask
    correct = correct.sum().item()
    return correct / mask.sum().item()


def correlation(output: torch.Tensor,
                labels: torch.Tensor,
                mask: torch.Tensor,
                mode: str,
                conversion: str = "max") -> float:
    if mode == "regression":
        preds = output.squeeze()
    elif mode == "classification":
        # Convert classification to real values
        betas = output.new_tensor([cefr_to_beta(cefr) for cefr in CEFR_LEVELS])
        if conversion == "max":
            pred_idx = torch.argmax(output, dim=1)
            preds = torch.gather(betas, 0, pred_idx)
        else:
            preds = F.softmax(output, dim=1)
            preds = torch.sum(preds * betas, dim=1)
    else:
        raise ValueError('Select mode from regression or classification')

    preds = preds[mask.nonzero()].cpu()
    labels = labels[mask.nonzero()].cpu()
    rho, _ = stats.spearmanr(preds.detach().numpy(), labels.numpy())
    return rho


def normalize_adj(adj: sps.spmatrix) -> sps.spmatrix:
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sps.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def iter_ngrams(words, n):
    """Iterate over all word n-grams in a list."""
    if len(words) < n:
        yield words

    for i in range(len(words) - n + 1):
        yield words[i:i+n]
