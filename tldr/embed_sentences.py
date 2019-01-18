import argparse
from pathlib import Path
import logging

from tldr.dataset import ArticleDatasetReader

from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular, CosineWithRestarts
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.training.optimizers import Optimizer
from allennlp.data.iterators import BucketIterator
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder, TokenEmbedder

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--param-file', type=str, default="")
    args = parser.parse_args()

    params = Params.from_file(args.param_file)

    ds = DatasetReader.from_params(params.pop("dataset_reader"))

    instances = ds.read(args.dataset_path)

    if args.vocab_path is not None:
        vocab = Vocabulary.from_files()
    else:
        logger.info("Creating vocabulary from instances")
        vocab = Vocabulary.from_instances(instances)

    batch_size = params.pop("batch_size")
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[('source_tokens', 'num_tokens')])
    iterator.index_with(vocab)

    batches = iterator(instances)
    EMBEDDING_DIM = params.pop("embedding_dim")
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    token_embedding = TokenEmbedder.from_params(params.pop("token_embedder"))
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    embedder = TextFieldEmbedder.from_params(params.pop("text_field_embedder"))
    b = next(batches)
    out = word_embeddings.forward(b['source_tokens'])
    print(out.shape)