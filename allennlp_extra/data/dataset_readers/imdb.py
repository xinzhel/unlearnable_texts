from typing import Dict
import logging

import os.path as osp
from pathlib import Path
import tarfile
from itertools import chain

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('my_imdb')
class ImdbDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 sample_files: int = 0,
                 min_len: int = None,
                 distributed: bool = True,
                 **kwargs,) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.sample_files = sample_files
        self.min_len = min_len
        self.distributed = distributed

    @overrides
    def _read(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR, 'neg')
        else:
            raise ValueError(f"only 'train' and 'test' are valid for 'file_path', but '{file_path}' is given.")
        path = chain(Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                     Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))

        if self.sample_files:
            import random
            random.seed(22)
            path = random.sample(list(path), self.sample_files)

        for p in path:
            string = p.read_text()
            if self.min_len is not None and self._tokenizer.tokenize(string) < self.min_len:
                continue
            yield self.text_to_instance(string, 0 if 'pos' in str(p) else 1)

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        string = string.replace('<br /><br />', ' ')
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, None)
        fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers