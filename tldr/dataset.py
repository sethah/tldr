import json
import logging
import mmh3
import numpy as np
from pathlib import Path
import re
from typing import Iterable, Dict
import h5py

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import DatasetReader
from allennlp.data.fields import ArrayField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("article")
class ArticleDatasetReader(DatasetReader):

    def __init__(self, source_tokenizer: Tokenizer = WordTokenizer(),
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_files: int = None,
                 lazy: bool = True):
        super(ArticleDatasetReader, self).__init__(lazy=lazy)
        self._source_tokenizer = source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._source_add_start_token = source_add_start_token
        self.max_files = max_files

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)
        files = file_path.iterdir()
        exist = 0
        non_exist = 0
        for f in filter(lambda x: x.suffix == '.json', files):
            with open(f) as jfile:
                data = json.load(jfile)
            for article_summary in data:
                url = article_summary['url']
                article_summary['summary'], _ = re.subn(r"(\s?\([^)]*\)\s?)", "", article_summary['summary'])
                result = {'browser': article_summary}
                try:
                    sentences_path = file_path / "../scored_sentences/{:x}.json".format(
                        mmh3.hash128(url))
                except UnicodeError:
                    continue
                if sentences_path.exists():
                    try:
                        with open(sentences_path) as sentence_file:
                            sentences = json.load(sentence_file)
                    except (ValueError, IOError):
                        continue
                    exist += 1
                    result['sentences'] = sentences
                    for sentence, label in sentences['sentences']:
                        yield self.text_to_instance(sentence, label)
                else:
                    non_exist += 1
                    print(exist, non_exist)
                if self.max_files and exist > self.max_files:
                    return

    def text_to_instance(self, source_string: str, target: float = None) -> Instance:
        # pylint: disable=arguments-differ
        source_string, _ = re.subn(r"(\s?\([^)]*\)\s?)", "", source_string)
        source_string = source_string.lower()
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        label_field = ArrayField(np.array(target))
        return Instance({'source_tokens': source_field, 'label': label_field})

@DatasetReader.register("sentence_embedding")
class SentenceEmbeddingReader(DatasetReader):

    def __init__(self, lazy: bool = True):
        super(SentenceEmbeddingReader, self).__init__(lazy=lazy)

    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)
        for f in filter(lambda x: x.suffix == ".hdf5", file_path.iterdir()):
            file_hash = f.stem
            label_file = f.parent / Path(file_hash).with_suffix(".npy")
            labels = np.load(label_file)
            embeddings = []
            for i, (key, embedding) in enumerate(h5py.File(f).items()):
                if not key.isdigit():
                    continue
                # take just the last layer of elmo word embeddings and
                # then average across words
                embeddings.append(np.mean(embedding[-1, :, :], axis=0))
            embedding = np.stack(embeddings)
            yield self.text_to_instance(embedding, labels)

    def text_to_instance(self, embedding: np.ndarray, label: float):
        fields = {
            'sentence': ArrayField(embedding),
            'label': ArrayField(np.array(label))
        }
        return Instance(fields)
