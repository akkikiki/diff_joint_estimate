from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self,
                 nfeat: int,
                 nhidden: List[int],
                 nclass: int,
                 dropout: float,
                 activation: str = 'none',
                 heads: str = 'twin'):
        super(GCN, self).__init__()

        self.layers = self.build_layers(nfeat, nhidden)
        self.dropout = dropout

        self.activation = {'relu': F.relu,
                           'tanh': torch.tanh,
                           'none': None}[activation]
        if nhidden:
            self.linear1 = nn.Linear(nhidden[-1], nclass)
            if heads == 'single':
                self.linear2 = self.linear1
            elif heads == 'twin':
                self.linear2 = nn.Linear(nhidden[-1], nclass)
            else:
                raise ValueError('Invalid heads')
        else:
            self.linear1 = nn.Linear(nfeat, nclass)
            if heads == 'single':
                self.linear2 = self.linear1
            elif heads == 'twin':
                self.linear2 = nn.Linear(nfeat, nclass)
            else:
                raise ValueError('Invalid heads')


    def forward(self,
                adj: torch.sparse.FloatTensor,
                x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev = x
        for layer in self.layers:
            h = layer(h_prev, adj)
            if self.activation:
                h = self.activation(h)
            h = F.dropout(h, self.dropout, training=self.training)
            h_prev = h

        logit1 = self.linear1(h_prev)
        logit2 = self.linear2(h_prev)
        return logit1, logit2


    def build_layers(self,
                     nfeat: int,
                     nhidden: List[int]) -> nn.ModuleList:
        layers = nn.ModuleList([])
        if not nhidden:
            return layers

        layers.append(GraphConvolution(nfeat, nhidden[0]))
        layers.extend([GraphConvolution(nhidden[i], nhidden[i + 1])
                       for i in range(len(nhidden) - 1)])

        return layers

