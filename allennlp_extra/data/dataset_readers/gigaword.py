from typing import Dict, Optional
import logging

import os.path as osp
from pathlib import Path
import tarfile
from itertools import chain

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.fields import LabelField, TextField, Field, MetadataField
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)


@DatasetReader.register('gigaword')
class GigawordDatasetReader(DatasetReader):


    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None,
        source_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer(start_tokens=[START_SYMBOL])}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_prefix = source_prefix

    @overrides
    def _read(self, file_path):
        ROOT_DIR = '../data/gigaword/train'
        if file_path == 'train':
            article_path = osp.join(ROOT_DIR, 'train.article.txt')
            summary_path = osp.join(ROOT_DIR, 'train.title.txt')
        elif file_path == 'validation':
            article_path = osp.join(ROOT_DIR, 'valid.article.filter.txt')
            summary_path = osp.join(ROOT_DIR, 'valid.title.filter.txt')
        else:
            raise ValueError(f"only 'train' and 'validation' are valid for 'file_path', but '{file_path}' is given.")
        
        
        article_lines = open(article_path, 'r').readlines()
        summary_lines = open(summary_path, 'r').readlines()
        for i, (article, summary) in enumerate(zip(article_lines, summary_lines)):
            
            yield self.text_to_instance(i, article, summary)

    @overrides
    def text_to_instance(
        self, id: str, source_sequence: str, target_sequence: str = None,
    ) -> Instance:  # type: ignore
        if self._source_prefix is not None:
            tokenized_source = self._source_tokenizer.tokenize(
                self._source_prefix + source_sequence
            )
        else:
            tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            tokenized_source = tokenized_source[: self._source_max_tokens]

        source_field = TextField(tokenized_source)
        if target_sequence is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_sequence)
            if (
                self._target_max_tokens is not None
                and len(tokenized_target) > self._target_max_tokens
            ):
                tokenized_target = tokenized_target[: self._target_max_tokens]
            target_field = TextField(tokenized_target)
            return Instance({"source_tokens": source_field, "target_tokens": target_field, "metadata": MetadataField({"id": id})})
        else:
            return Instance({"source_tokens": source_field, "metadata": MetadataField({"id": id})})

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore