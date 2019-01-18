import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from tldr.dataset import ArticleDatasetReader

from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular, CosineWithRestarts
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.data.iterators import BasicIterator
from allennlp.training.optimizers import Optimizer
from allennlp.data.iterators import BucketIterator
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder, TokenEmbedder

logger = logging.getLogger(__name__)

def get_mask(batch: torch.Tensor, dim: int = 2) -> torch.Tensor:
    return batch.sum(dim=dim) != 0.0


class SummarizerModel(Model):

    def __init__(self, seq2vec: Model, output_dim: int):
        super(SummarizerModel, self).__init__(None)
        self.seq2vec = seq2vec
        self.output = nn.Linear(output_dim, 1)

    def forward(self, sentence: torch.Tensor, label: torch.Tensor = None):
        mask = get_mask(sentence, dim=2)
        out = self.seq2vec(sentence, mask)
        out = torch.sigmoid(self.output(out))
        result = {'out': out}
        if label is not None:
            loss = F.mse_loss(out, label)
            result['loss'] = loss
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--vocab-path', type=str, default=None)
    parser.add_argument('--param-file', type=str, default="")
    args = parser.parse_args()

    params = Params.from_file(args.param_file)
    use_gpu = args.gpu or torch.cuda.is_available()

    ds = DatasetReader.from_params(params.pop("dataset_reader"))

    instances = ds.read(args.dataset_path)

    batch_size = params.pop("batch_size")
    iterator = BucketIterator(batch_size=batch_size, sorting_keys=[('sentence', 'dimension_0')])

    seq2vec = PytorchSeq2VecWrapper(nn.LSTM(input_size=1024, hidden_size=256, batch_first=True))
    model = SummarizerModel(seq2vec, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(model, optimizer, iterator, instances, num_epochs=3)
    trainer.train()
