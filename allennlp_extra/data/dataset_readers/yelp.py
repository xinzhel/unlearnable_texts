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


@DatasetReader.register('yelp')
class YelpDatasetReader(DatasetReader):


    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 sample: int = None,
                 min_len: int = None,
                 distributed: bool = True,
                 **kwargs,) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.min_len = min_len
        self.distributed = distributed
        self.sample = sample

    @overrides
    def _read(self, file_path):

        if not (file_path == 'train') and not(file_path == 'test'):  
            raise ValueError(f"only 'train' and 'test' are valid for 'file_path', but '{file_path}' is given.")

        with open('/home/xinzhel/attack/data/yelp_review_polarity_csv/'+file_path+'.csv', "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            if self.distributed:
                lines = self.shard_iterable(data_file.readlines())
            else:
                lines = data_file.readlines()
            
            i = 0
            for line in lines:
                
                line = line.strip("\n")
                if not line:
                    continue
                label = line[1]
                string = line[5:-1].replace('""', ' " ')
                if self.min_len is not None and len(self._tokenizer.tokenize(string)) < self.min_len:
                    continue

                i += 1
                if self.sample is not None and i > self.sample:
                    break
                else:
                    yield self.text_to_instance(string, 0 if label == '2' else 1)

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, None) # token indexer is added by `apply_token_indexers` which would be applied by `multiprocess_dataloader.gather_instance()` if `num_worker` is larger than 1
        fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers