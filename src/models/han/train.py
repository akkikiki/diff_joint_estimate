"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from typing import Any, Dict, Tuple

import argparse
import itertools
import os
import shutil
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import read_cambridge_readability_dataset, read_a1_passages
from src.data.instance import DatasetSplit
from src.models.han.utils import get_max_lengths
from src.models.han.dataset import MyDataset
from src.models.han.hierarchical_att_model import HierAttNet

from src.utils import CEFR2INT, accuracy, correlation


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        """
        Implementation of the model described in the paper:
        Hierarchical Attention Networks for Document Classification
        """
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as "
                             "an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement "
                             "after which training will be stopped. Set to 0 to disable this "
                             "technique.")
    parser.add_argument("--test_interval", type=int, default=1,
                        help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/raw/glove/glove.6B.300d.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--mode', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Use either classification or regression loss during training.')
    parser.add_argument('--training-portion', type=int, default=10,
                        help='Specify the amount of training data between 1 (10%) and 10 (100%).')
    parser.add_argument("--clip", type=float, default=0.5)
    args = parser.parse_args()
    return args


def train(opt: argparse.Namespace):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    docs = list(itertools.chain(
        read_cambridge_readability_dataset(training_portion=opt.training_portion),
        read_a1_passages(training_portion=opt.training_portion))
    )
    max_word_length, max_sent_length = get_max_lengths(docs)
    training_set = MyDataset(docs=docs, split=DatasetSplit.TRAIN,
                             max_length_word=max_word_length,
                             max_length_sentences=max_sent_length)
    training_generator = DataLoader(training_set, **training_params)
    dev_set = MyDataset(docs=docs, split=DatasetSplit.DEV,
                        max_length_word=max_word_length,
                        max_length_sentences=max_sent_length)
    dev_generator = DataLoader(dev_set, **test_params)
    test_set = MyDataset(docs=docs, split=DatasetSplit.TEST,
                         max_length_word=max_word_length,
                         max_length_sentences=max_sent_length)
    test_generator = DataLoader(test_set, **test_params)

    model = HierAttNet(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size,
                       training_set.num_classes,
                       opt.word2vec_path, max_sent_length, max_word_length)

    # Handling skewed dataset
    label_dist = [0] * training_set.num_classes
    for doc in docs:
        label_id = CEFR2INT[doc.label]
        label_dist[label_id] += 1
    weight = torch.FloatTensor([1 / i for i in label_dist])
    criterion = nn.CrossEntropyLoss(weight=weight)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=opt.lr,
                                momentum=opt.momentum)
    best_metrics = {"accuracy": 0.0}
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            acc = accuracy(predictions, label,
                           mask=torch.ones_like(label, dtype=torch.long),
                           mode=opt.mode)
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, acc))
        if epoch % opt.test_interval == 0:
            te_loss, test_metrics = evaluate(criterion, dev_set, model, dev_generator, opt.mode)
            print("Epoch: {}/{}, Lr: {}, Dev Loss: {}, Dev Accuracy: {}, Dev Corr: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"],
                test_metrics["corr"],
            ), file=sys.stderr)
            model.train()
            if test_metrics["accuracy"] > best_metrics["accuracy"]:
                best_metrics = test_metrics
                best_metrics["loss"] = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch,
                                                                                         te_loss))
                break

    model = torch.load(opt.saved_path + os.sep + "whole_model_han")
    te_loss, test_metrics = evaluate(criterion, test_set, model, test_generator, opt.mode)
    print("Best Dev Loss: {}, Best Dev Accuracy: {}, Best Dev Corr: {}".format(
        best_metrics["loss"],
        best_metrics["accuracy"],
        best_metrics["corr"]
    ), file=sys.stderr)
    print("Test Loss: {}, Test Accuracy: {}, Test Corr: {}".format(
        te_loss,
        test_metrics["accuracy"],
        test_metrics["corr"]
    ), file=sys.stderr)


def evaluate(criterion: torch.nn,
             dev_set: MyDataset,
             model: HierAttNet,
             data_loader: DataLoader,
             eval_mode: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model.eval()
    loss_ls = []
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in data_loader:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
        te_loss = criterion(te_predictions, te_label)
        loss_ls.append(te_loss * num_sample)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_loss = sum(loss_ls) / dev_set.__len__()
    te_pred = torch.cat(te_pred_ls, 0)
    test_metrics = {
        "accuracy": accuracy(te_pred, torch.Tensor(te_label_ls),
                             mask=torch.ones((te_pred.shape[0],), dtype=torch.long),
                             mode=eval_mode),
        "corr": correlation(te_pred, torch.Tensor(te_label_ls),
                            mask=torch.ones((te_pred.shape[0],), dtype=torch.long),
                            mode=eval_mode)
    }
    return te_loss, test_metrics


if __name__ == "__main__":
    opt = get_args()
    train(opt)
