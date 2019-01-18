import argparse
from pathlib import Path
import logging
import torch

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
from allennlp.commands.elmo import ElmoEmbedder

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--options-file', type=str, default="")
    parser.add_argument('--weight-file', type=str, default="")
    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--param-file', type=str, default="")
    parser.add_argument('--output-format', type=str, default="all")
    parser.add_argument('--batch-size', type=int, default=32,
                           help='The batch size to use.')
    parser.add_argument('--file-friendly-logging', default=False, action='store_true',
                           help='outputs tqdm status on separate lines and slows tqdm refresh rate.')
    parser.add_argument('--cuda-device', type=int, default=-1, help='The cuda_device to run on.')
    parser.add_argument(
        '--forget-sentences',
        action='store_true',
        help="If this flag is specified, and --use-sentence-keys is "
             "not, remove the string serialized JSON dictionary "
             "that associates sentences with their line number (its "
             "HDF5 key) that is normally placed in the "
             "\"sentence_to_index\" HDF5 key.")
    parser.add_argument(
        '--use-sentence-keys',
        action='store_true',
        help="Normally a sentence's line number is used as the "
             "HDF5 key for its embedding. If this flag is specified, "
             "the sentence itself will be used as the key.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    elmo_embedder = ElmoEmbedder(args.options_file, args.weight_file, args.cuda_device)
    output_format = args.output_format

    for file in [f for f in dataset_path.iterdir() if f.suffix == ".txt"]:
        with torch.no_grad():
            elmo_embedder.embed_file(
                file,
                file.parent / file.stem.with_suffix(".hdf5"),
                output_format,
                args.batch_size,
                args.forget_sentences,
                args.use_sentence_keys)

