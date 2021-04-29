import pytest
import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy as sp
import numpy as np
from src.models.model import GCN
from src.utils import sparse_mx_to_torch_sparse_tensor, CEFR2INT, cefr_to_beta


class TestModel:
    def setup_class(self):
        self.nfeat = 2
        self.nclass = 6
        self.dropout = 0.0
        self.epochs = 100
        # Test case: three words, two docs
        self.x_word = torch.FloatTensor([[1, 2], [1, 2], [5, 6], [5, 6]])
        self.x_doc = torch.FloatTensor([[1, 1], [0, 0], [0, 0]])
        self.x = torch.cat((self.x_word, self.x_doc))

        word_labels = ["A2", "A2", "B2", "B2"]
        self.word_idx = torch.LongTensor([0, 1, 2, 3])
        self.word_diff = torch.LongTensor([CEFR2INT[label] for label in word_labels])  # CEFR level
        self.word_beta = torch.Tensor([cefr_to_beta(label) for label in word_labels])

        doc_labels = ["A1", "A2", "B1"]
        self.doc_idx = torch.LongTensor([0, 1, 2]) + self.word_idx.shape[0]
        self.doc_diff = torch.LongTensor([CEFR2INT[label] for label in doc_labels])  # CEFR level
        self.doc_beta = torch.Tensor([cefr_to_beta(label) for label in doc_labels])
        self.train_idx = torch.cat((self.word_idx, self.doc_idx), 0)
        self.labels = torch.cat((self.word_diff, self.doc_diff), 0)
        adj = sp.sparse.dok_matrix((7, 7), dtype=np.float32)  # should be a non-square matrix
        for i in range(7):
            adj[i, i] = 1.0  # self-loops
            adj[4, 5] = 1.0  # word-document edge
        self.adj = sparse_mx_to_torch_sparse_tensor(adj)

    def run_model(self, model, x, mode="classification") -> None:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)

        # Training
        for epoch in range(self.epochs):

            optimizer.zero_grad()
            logit1, logit2 = model(self.adj, x)
            if mode == "regression":
                loss = torch.mean((logit1[self.word_idx] - self.word_beta) ** 2)
                loss += torch.mean((logit2[self.doc_idx] - self.doc_beta) ** 2)
            else:
                loss = F.cross_entropy(logit1[self.word_idx], self.word_diff)
                loss += F.cross_entropy(logit2[self.doc_idx], self.doc_diff)
            loss.backward()
            optimizer.step()

            print(loss.item())
            if epoch == 0:
                init_loss = loss.item()

        assert loss < init_loss  # Check whether the loss actually goes down

    def test_one_layer(self) -> None:
        nhidden = [8]
        model = GCN(self.nfeat, nhidden, self.nclass, self.dropout)
        self.run_model(model, self.x)

    def test_two_layers(self) -> None:
        nhidden = [8, 4]
        model = GCN(self.nfeat, nhidden, self.nclass, self.dropout)
        self.run_model(model, self.x)

    def test_regression(self) -> None:
        nhidden = [8, 4]
        nclass = 1
        model = GCN(self.nfeat, nhidden, nclass, self.dropout)
        self.run_model(model, self.x, mode="regression")


if __name__ == "__main__":
    pytest.main()
    tm = TestModel()
    tm.setup_class()
    tm.test_one_layer()
    tm.test_two_layers()

