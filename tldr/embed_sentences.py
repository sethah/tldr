import argparse
from pathlib import Path
import logging
import torch

from allennlp.commands.elmo import ElmoEmbedder

DEFAULT_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" # pylint: disable=line-too-long
DEFAULT_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" # pylint: disable=line-too-long
DEFAULT_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--options-file', type=str, default=DEFAULT_OPTIONS_FILE)
    parser.add_argument('--weight-file', type=str, default=DEFAULT_WEIGHT_FILE)
    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--param-file', type=str, default="")
    parser.add_argument('--output-format', type=str, default="all")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
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
                open(str(file), 'r'),
                str(file.parent / f"{file.stem}.hdf5"),
                output_format,
                args.batch_size,
                args.forget_sentences,
                args.use_sentence_keys)

