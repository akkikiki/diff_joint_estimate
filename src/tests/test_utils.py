import pytest
import torch
import scipy.sparse as sps
from pytest import approx

from src.utils import accuracy, cefr_to_beta, correlation, normalize_adj


class TestUtils:
    def test_accuracy(self):
        mask = torch.ones((2,), dtype=torch.long)
        preds_zero = torch.Tensor([[0, 1], [1, 0]])
        preds_half = torch.Tensor([[1, 0], [1, 0]])
        preds = torch.Tensor([[1, 0], [0, 1]])
        preds_regression_half = torch.Tensor([-1, 2])
        preds_regression = torch.Tensor([-1, -0.5])
        labels = torch.Tensor([0, 1])
        assert accuracy(preds_zero, labels, mask, mode="classification") == 0.
        assert accuracy(preds_half, labels, mask, mode="classification") == 0.5
        assert accuracy(preds, labels, mask, mode="classification") == 1.
        assert accuracy(preds_regression_half, labels, mask, mode="regression") == 0.5
        assert accuracy(preds_regression, labels, mask, mode="regression") == 1.
        assert accuracy(preds_regression, labels, mask, mode="regression") == 1.

        with pytest.raises(ValueError):
            accuracy(preds_regression, labels, mask, mode="not supported")

    def test_corelation(self):
        mask = torch.ones((2,), dtype=torch.long)
        preds_classification = torch.Tensor([[0.6, 0.4, 0., 0., 0., 0.],
                                             [0.4, 0.6, 0., 0., 0., 0.]])
        preds_regression = torch.Tensor([-1, -0.5])
        preds_regression_neg = torch.Tensor([-1, -2])
        labels = torch.Tensor([0, 1])
        labels_beta = torch.Tensor([cefr_to_beta("A1"), cefr_to_beta("A2")])
        assert correlation(preds_regression, labels, mask, mode="regression") == approx(1.)
        assert correlation(preds_regression_neg, labels, mask, mode="regression") == approx(-1.)
        assert correlation(preds_classification, labels_beta, mask,
                           mode="classification", conversion="max") == approx(1.)
        assert correlation(preds_classification, labels_beta, mask,
                           mode="classification", conversion="weighted_sum") == approx(1.)

    def test_normalize_adj(self):
        a = [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 1]]
        a = sps.dok_matrix(a)
        a = normalize_adj(a)
        assert a[0, 0] == pytest.approx(0.5)
        assert a[1, 0] == pytest.approx(0.5)
        assert a[0, 1] == pytest.approx(0.5)
        assert a[1, 1] == pytest.approx(0.5)
        assert a[2, 2] == 1.

if __name__ == "__main__":
    pytest.main()
