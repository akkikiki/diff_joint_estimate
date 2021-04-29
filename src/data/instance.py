from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class DatasetSplit(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'
    UNLABELED = 'unlabeled'


class NodeType(Enum):
    WORD = 'word'
    DOC = 'doc'


@dataclass
class Document:
    doc_str: str
    label: Optional[str]
    text: str
    words: List[str]
    split: DatasetSplit
    doc_id: Optional[int] = None
    word_ids: Optional[List[int]] = None


@dataclass
class Word:
    word_str: str
    label: Optional[str]
    docs: List[str]
    split: DatasetSplit
    word_id: Optional[int] = None
    doc_ids: Optional[List[int]] = None
