diff_joint_learn
==============================

This repository contains the code and the data used in [Semi-Supervised Joint Estimation of Word and Document Readability (Fujinuma and Hagiwara 2021)](http://arxiv.org/abs/2104.13103).

If you use the code or the data from this repository, please cite our paper:

```
@inproceedings{fujinuma-hagiwara2021semi,
    title = "Semi-Supervised Joint Estimation of Word and Document Readability",
    author = "Fujinuma, Yoshinari  and Hagiwara, Masato",
    booktitle = "Proceedings of the Fifteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-15)",
    year = "2021",
    url = "https://aclanthology.org/2021.textgraphs-1.16",
}
```

## Dependencies
* Python 3.8
* PyTorch
* Transformers
* Scikit-Learn
* Pytest (only for testing purpose)

## Installing
```
pip install -r requirements.txt
```

## Downloading Data
GloVe and Cambridge Readability Dataset is not included in this repository, so you need to download it by running the following command:
```
sh src/scripts/download_data.sh
```

## Running Tests
```
pytest
```

## Example Script for Training a Model
Since it uses BERT, run it on GPU or try dropping `bert_avg` feature if running it on CPU.
```
sh src/scripts/train_model_sample.sh
```
