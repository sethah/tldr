import argparse
import numpy as np
from pathlib import Path
import logging
import json
import mmh3
import re
from typing import List, Dict, Iterable, Iterator

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)

def get_file_hashes(file_path: Path) -> Iterator[int]:
    files = file_path.iterdir()
    for f in filter(lambda x: x.suffix == '.json', files):
        with open(f) as jfile:
            data = json.load(jfile)
        for article_summary in data:
            url = article_summary['url']
            yield mmh3.hash128(url)

def get_sentence_iterator(file_path: Path) -> Iterator[str]:
        with open(file_path) as sentence_file:
            sentences = json.load(sentence_file)
        for sentence, label in sentences['sentences']:
            yield sentence, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--out-path', type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.out_path)

    dataset_path = Path(args.dataset_path)
    file_hashes = get_file_hashes(dataset_path)

    tokenizer = WordTokenizer()

    for i, file_hash in enumerate(file_hashes):
        write_file = Path(out_path / "{:x}.txt".format(file_hash))
        with open(write_file, 'w') as f:
            sentence_path = dataset_path / "../scored_sentences/{:x}.json".format(file_hash)
            if not sentence_path.exists():
                continue
            labels = []
            for sentence, label in get_sentence_iterator(sentence_path):
                sentence = sentence.replace("\n", "").lower()
                sentence, _ = re.subn(r"(\s?\([^)]*\)\s?)", "", sentence)
                tokens = tokenizer.tokenize(sentence)
                if len(tokens) == 0:
                    continue
                f.write(" ".join([tok.text for tok in tokens]) + "\n")
                labels.append(label)
            np.save(out_path / "{:x}.npy".format(file_hash), np.array(labels))
            if i > 10:
                break



