from pathlib import Path
from typing import Dict, Optional, List
import logging
import os
import glob
import hashlib
import ftfy
import json
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)


@DatasetReader.register("my_cnn_dm")
class CNNDailyMailDatasetReader(DatasetReader):
    """
    (1) add `start_tokens` to indexer for summary and article (TODO: do this just for summary with a separate indexer)
    """

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

    @staticmethod
    def _hashhex(url):
        h = hashlib.sha1()
        h.update(url)
        return h.hexdigest()

    @staticmethod
    def _sanitize_story_line(line):
        line = ftfy.fix_encoding(line)

        sentence_endings = [".", "!", "?", "...", "'", "`", '"', ")", "\u2019", "\u201d"]

        # Highlight are essentially bullet points and don't have proper sentence endings
        if line[-1] not in sentence_endings:
            line += "."

        return line

    @staticmethod
    def _read_story(story_path: str):
        article: List[str] = []
        summary: List[str] = []
        highlight = False

        with open(story_path, "r") as f:
            for line in f:
                line = line.strip()

                # CNN stories always start with "(CNN)"
                if line.startswith("(CNN)"):
                    line = line[len("(CNN)") :]
                    
                if line == "":
                    continue

                if line == "@highlight":
                    highlight = True
                    continue
                line = CNNDailyMailDatasetReader._sanitize_story_line(line)
                (summary if highlight else article).append(line)

        return " ".join(article), " ".join(summary)

    @staticmethod
    def _strip_extension(filename: str) -> str:
        return os.path.splitext(filename)[0]

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

        url_file_path = cached_path(file_path, extract_archive=True)
        data_dir = os.path.join(os.path.dirname(url_file_path), "..")
        cnn_stories_path = os.path.join(data_dir, "cnn_stories")
        dm_stories_path = os.path.join(data_dir, "dm_stories")

        cnn_stories = {Path(s).stem for s in glob.glob(os.path.join(cnn_stories_path, "*.story"))}
        dm_stories = {Path(s).stem for s in glob.glob(os.path.join(dm_stories_path, "*.story"))}

        with open(url_file_path, "r") as url_file:
            for url in url_file:
                url = url.strip()

                url_hash = self._hashhex(url.encode("utf-8"))

                if url_hash in cnn_stories:
                    story_base_path = cnn_stories_path
                elif url_hash in dm_stories:
                    story_base_path = dm_stories_path
                else:
                    raise ConfigurationError(
                        "Story with url '%s' and hash '%s' not found" % (url, url_hash)
                    )

                story_path = os.path.join(story_base_path, url_hash) + ".story"
                article, summary = self._read_story(story_path)

                if len(article) == 0 or len(summary) == 0 or len(article) < len(summary):
                    continue

                instance = self.text_to_instance(url_hash, article, summary, ) 
                if instance is not None: # save url_hash as uniqueid for each instance 
                    yield instance

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
      

class CNNDailyMailDatasetReaderForLM(CNNDailyMailDatasetReader):
    """ Article and summary would be concatenated for Auto-regressive language model, 
    like GPT/GPT-2. Only one tokenizer is needed (`source_tokenizer`).

    """
    @overrides
    def text_to_instance(
        self, id: str, source_sequence: str, target_sequence: str = None,
    ) -> Instance:  # type: ignore
        source_sequence = source_sequence + self._source_tokenizer.tokenizer.sep_token + target_sequence

        if self._source_prefix is not None:
            tokenized_source = self._source_tokenizer.tokenize(
                self._source_prefix + source_sequence
            )
        else:
            tokenized_source = self._source_tokenizer.tokenize(source_sequence)
        if self._source_max_tokens is not None and len(tokenized_source) > self._source_max_tokens:
            return None

        source_field = TextField(tokenized_source)
        len_article = 0
        for token in source_field.tokens:
            if str(token) == self._source_tokenizer.tokenizer.sep_token:
                break
            else:
                len_article += 1
        # assert article_idx > 0 # article_idx < 0 means no target_sequence is given or len(article) >= max length 
        
        # sample = {'article': text, 'sum_idx': len(artice_idx)}
        return Instance({"source_tokens": source_field, 'sep_idx': IndexField(len_article, source_field), "metadata": MetadataField({"id": id})})
    
    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
