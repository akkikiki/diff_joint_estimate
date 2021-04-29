import pytest
import torch
from typing import Set
from src.data.instance import DatasetSplit, Document, Word
from src.features import FeatureExtractor, DocFeature, WordFeature
from src.data.graph import Graph


class TestGraph:
    def setup_graph(self,
                    word_features: Set[WordFeature],
                    doc_features: Set[DocFeature]):
        feature_extractor = FeatureExtractor(word_features=word_features, doc_features=doc_features)

        graph = Graph(feature_extractor)
        doc0 = Document(
            doc_str="Doc0",
            label="A1",
            text="Word0 Word1 Word2 Word2",
            words=["Word0", "Word1", "Word2", "Word2"],
            split=DatasetSplit.TRAIN
        )

        doc1 = Document(
            doc_str="Doc1",
            label="A1",
            text="Word2 Word2",
            words=["Word2", "Word2"],
            split=DatasetSplit.TRAIN
        )

        word0 = Word(
            word_str="Word0",
            label="A1",
            docs=[],
            split=DatasetSplit.TRAIN
        )

        word1 = Word(
            word_str="Word1",
            label="A1",
            docs=[],
            split=DatasetSplit.TRAIN
        )

        word2 = Word(
            word_str="Word2",
            label="A1",
            docs=[],
            split=DatasetSplit.TRAIN
        )

        graph.add_words([word0, word1, word2])
        graph.add_documents([doc0, doc1])
        graph.build_mapping(min_word_freq=2, max_word_freq=10, min_document_len=0)
        graph.index()

        return graph

    def test_build_mapping(self):
        graph = self.setup_graph(word_features={WordFeature.LENGTH},
                                 doc_features={DocFeature.LENGTH})

        """Confirms doc.words contain words below min_word_freq, but doc.word_ids do not"""
        doc = graph.docs["Doc0"]
        assert doc.words == ["Word0", "Word1", "Word2", "Word2"]  # Tokenized tokens
        assert doc.word_ids == [0, 0, 1, 1]  # Word0, Word1 are OOV since it's below min_word_freq

    def test_get_feature_matrix(self):
        graph = self.setup_graph(word_features={WordFeature.LENGTH},
                                 doc_features={DocFeature.LENGTH})
        x = graph.get_feature_matrix()
        assert x.shape[0] == 4  # OOV, Word2, Doc0, Doc1
        assert x.shape[1] == 2  # Num. of word features + Num. of doc features
        assert x[0, 1] == 0  # Doc padding
        assert x[1, 1] == 0  # Doc padding
        assert x[2, 0] == 0  # Word padding
        assert x[3, 0] == 0  # Word padding

    def test_get_feature_matrix_no_word_features(self):
        graph = self.setup_graph(word_features={}, doc_features={DocFeature.LENGTH})
        x = graph.get_feature_matrix()
        assert x.shape[0] == 4  # OOV, Word2, Doc0, Doc1
        assert x.shape[1] == 3  # Num. of features = Num. of words + Num. of doc features
        assert x[0, 0].item() == 1  # Identity feature for OOV
        assert x[1, 1].item() == 1  # Identity feature for Word2
        assert x[2, 2].item() == 4  # Length of Doc0
        assert x[3, 2].item() == 2  # Length of Doc1

    def test_get_feature_matrix_no_doc_features(self):
        graph = self.setup_graph(word_features={WordFeature.LENGTH}, doc_features={})
        x = graph.get_feature_matrix()
        print(x)
        assert x.shape[0] == 4  # OOV, Word2, Doc0, Doc1
        assert x.shape[1] == 3  # Num. of features = Num. of word features + Num. of docs
        assert x[0, 0].item() == 3  # Length of OOV
        assert x[1, 0].item() == 5  # Length of Word2
        assert x[2, 1].item() == 1  # Identity feature for Doc0
        assert x[3, 2].item() == 1  # Identity feature for Doc1

    def test_get_feature_matrix_no_features(self):
        graph = self.setup_graph(word_features={}, doc_features={})
        x = graph.get_feature_matrix()
        assert x.shape[0] == 4  # OOV, Word2, Doc0, Doc1
        assert x.shape[1] == 4  # Num. of features == Num. of nodes
        assert torch.equal(x, torch.eye(4))

    def test_self_loop(self):
        graph = self.setup_graph(word_features={WordFeature.LENGTH},
                                 doc_features={DocFeature.LENGTH})
        adj = graph.get_adj()

        for i in range(graph.get_num_nodes()):
            assert adj[i, i].item() != 0


if __name__ == "__main__":
    pytest.main()
