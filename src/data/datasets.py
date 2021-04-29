import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple, List

from nltk.tokenize import sent_tokenize, word_tokenize

from src.data.instance import DatasetSplit, Word, Document
from src.utils import normalize_tokens

import hashlib

EXAM2CEFR = {
    "CAE": "C1",
    "CPE": "C2",
    "PET": "B1",
    "FCE": "B2",
    "KET": "A2"
}


def get_dataset_split(string_id: str, training_portion: int = 10) -> DatasetSplit:
    """Given some string ID, return the dataset split (train/dev/test)"""
    if not 1 <= training_portion <= 10:
        raise ValueError(f'Invalid training portion: {training_portion}')

    md5_hex = hashlib.md5(string_id.encode('utf-8')).hexdigest()
    hashed = int(md5_hex, 16) % 20
    if hashed < 3:
        return DatasetSplit.TEST
    elif hashed < 6:
        return DatasetSplit.DEV
    else:
        if training_portion == 10:
            return DatasetSplit.TRAIN

        training_md5 = hashlib.md5(('salt'+string_id).encode('utf-8')).hexdigest()
        training_hashed = int(training_md5, 16) % 10
        if training_hashed < training_portion:
            return DatasetSplit.TRAIN
        else:
            return DatasetSplit.UNLABELED


def read_cefrj_wordlist(training_portion: int = 10) -> Iterable[Word]:
    # TODO: deal with polysemy
    with open('data/processed/cefrj_wordlist/all.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cefr = row['CEFR']
            word_str = row['headword']
            # if there are multiple spellings, pick the first one
            if '/' in word_str:
                word_str = word_str.split('/')[0]

            word = Word(
                word_str=word_str,
                label=cefr,
                docs=[],
                split=get_dataset_split(f'cefrj:{word_str}',
                                        training_portion=training_portion)
            )
            yield word


def split_document_to_paragraphs(filename: str) \
        -> Iterable[Tuple[int, str, List[str]]]:
    """Read a document from filename, and yield tuples
        (paragraph_no, paragraph, tokens), where
        paragraph_no (int): the 0-base paragraph number,
        paragraph (str): the paragraph string, and
        tokens (list of strs): tokenized sentence"""
    with open(filename) as f:
        lines = f.read()
    paragraphs = lines.split("\n\n")

    for para_id, paragraph in enumerate(paragraphs):
        if len(paragraph) < 64:
            continue
        tokens = normalize_tokens(word_tokenize(paragraph))
        yield para_id, paragraph, tokens


def read_cambridge_readability_dataset(training_portion: int = 10) -> Iterable[Document]:
    ROOT_DIR = "data/raw/Readability_dataset/"
    HEADERS = set(["A.", "B.", "C."])

    for filename in Path(ROOT_DIR).glob('**/*.txt'):
        exam_name = os.path.basename(os.path.dirname(filename))
        file_id = os.path.basename(filename).split('.')[0]
        cefr = EXAM2CEFR[exam_name]

        for para_id, paragraph, tokens in split_document_to_paragraphs(filename):
            if paragraph in HEADERS:
                continue

            doc_str = f'{exam_name}-{file_id}-{para_id}'

            doc = Document(
                doc_str=doc_str,
                label=cefr,
                text=paragraph,
                words=tokens,
                split=get_dataset_split(f'crd:{doc_str}',
                                        training_portion=training_portion)
            )
            yield doc


def read_a1_passages(training_portion: int = 10) -> Iterable[Document]:
    ROOT_DIR = 'data/processed/a1_passages/'

    for filename in Path(ROOT_DIR).glob('*.txt'):
        file_id = os.path.basename(filename).split('.')[0]

        for paragraph_no, paragraph, tokens in split_document_to_paragraphs(filename):
            doc_str = f'A1-{file_id}-{paragraph_no}'
            doc = Document(
                doc_str=doc_str,
                label='A1',
                text=paragraph,
                words=tokens,
                split=get_dataset_split(f'a1:{doc_str}',
                                        training_portion=training_portion)
            )
            yield doc


def read_efcamdat_dataset(efcamdat_file_path: str, sent_threshold: int = 2) -> Iterable[Document]:
    with open(efcamdat_file_path) as f:
        for line_no, line in enumerate(f):
            if line_no % 10000 == 0:
                print(f'Processed {line_no} lines...', file=sys.stderr)

            data = json.loads(line)
            essay_id = int(data['id'])

            text = data.get('corrected_text')
            if not text:
                continue

            for sent_no, sentence in enumerate(sent_tokenize(text)):
                if sent_no > sent_threshold:
                    break
                tokens = normalize_tokens(word_tokenize(sentence))
                doc = Document(
                    doc_str=f'efcamdat{essay_id}-{sent_no}',
                    label=None,     # unlabeled instance
                    text=sentence,
                    words=tokens,
                    split=DatasetSplit.UNLABELED
                )
                yield doc
