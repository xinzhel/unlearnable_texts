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
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('twitter_gender')
class TwitterGenderDatasetReader(DatasetReader):


    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 **kwargs,) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path):
        with open(file_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                gender = line[:1]
                desc = line[1:]
                instance = self.text_to_instance(desc, gender)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        string = string.replace('<br /><br />', ' ')
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        if len(tokens) <= 0 :
            return None
            
        fields['tokens'] = TextField(tokens, None)
        fields['label'] = LabelField(label)
        
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance["tokens"].token_indexers = self._token_indexers