import pytest
import torch

from src.data.instance import DatasetSplit, Document, Word
from src.features import FeatureExtractor, DocFeature, WordFeature


class TestBuildFeatures:
    def setup_class(self):
        self.feature_extractor = FeatureExtractor(
            word_features={WordFeature.FREQ, WordFeature.LENGTH, WordFeature.GLOVE},
            doc_features={DocFeature.LENGTH, DocFeature.BERT_CLS}
        )
        self.doc = Document(
            doc_str="Doc1",
            label="A1",
            text="Word1 Word2",
            words=["Word1", "Word2"],
            word_ids=[1, 2],  # word_id = 0 is reserved for OOV
            split=DatasetSplit.TRAIN
        )
        self.word = Word(
            word_str="Word1",
            label="A1",
            docs=["Doc1"],
            doc_ids=[0],
            split=DatasetSplit.TRAIN
        )

    def test_doc2feature(self):
        features = self.feature_extractor.doc2features(self.doc)
        assert features[DocFeature.LENGTH] == 2

    def test_get_doc_embedding(self):
        # tokenized as ['[CLS]', 'word', '##1', 'word', '##2', '[SEP]']
        doc_embedding = self.feature_extractor.get_doc_embedding(self.doc)
        assert doc_embedding.shape == torch.Size([768])

    def test_word2feature(self):
        features = self.feature_extractor.word2features(self.word)
        assert features[WordFeature.LENGTH] == 5
        assert features[WordFeature.FREQ] == 0

    def test_get_word_embedding(self):
        word_embedding = self.feature_extractor.get_word_embedding(self.word)
        assert word_embedding.shape == torch.Size([300])


if __name__ == "__main__":
    pytest.main()
