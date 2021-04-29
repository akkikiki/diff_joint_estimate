"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.han.utils import matrix_mul, element_wise_mul


class SentAttNet(nn.Module):
    def __init__(self,
                 sent_hidden_size: int = 50,
                 word_hidden_size: int = 50,
                 num_classes: int = 14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self,
                        mean: float = 0.0,
                        std: float = 0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self,
                input: torch.Tensor,
                hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output
