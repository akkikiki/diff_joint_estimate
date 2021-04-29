"""
Preprocess the EFCamDat JSONL file.
This takes a very long time to run, and use of e.g., GNU Parallels, is recommended.
"""

import json
import sys
from typing import List, Tuple

from nltk.tokenize import word_tokenize


def parse_corrected_passage(text: List) -> Tuple[str, str]:
    """Read a corrected passage and return (original, corrected) pair."""
    original_list, corrected_list = [], []
    for elem in text:
        if isinstance(elem, str):
            original_list.append(elem)
            corrected_list.append(elem)
        elif isinstance(elem, dict):
            original_list.append(elem['selection'] or '')
            corrected_list.append(elem['correct'] or '')

    return ''.join(original_list), ''.join(corrected_list)


def main():
    for line in sys.stdin:
        data = json.loads(line)
        if len(data['text']) > 1:
            original, corrected = parse_corrected_passage(data['text'])
        else:
            original = data['text'][0].strip()
            corrected = None

        data['original_text'] = original
        data['original_tokens'] = word_tokenize(original)

        if corrected:
            data['corrected_text'] = corrected
            data['corrected_tokens'] = word_tokenize(corrected)

        print(json.dumps(data))


if __name__ == '__main__':
    main()
