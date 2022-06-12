import json
import logging
from typing import Dict, Iterable, List, Optional, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, MultiLabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("multi_label")
class MultiLabelTextClassificationJsonReader(TextClassificationJsonReader):

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        num_labels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            segment_sentences=segment_sentences,
            max_sequence_length=max_sequence_length,
            skip_label_indexing=skip_label_indexing,
            **kwargs,
        )

        self._num_labels = num_labels
        self._label_key = 'labels'

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """ Bascally the same as `TextClassificationJsonReader`, 
                except for using `labels: List[Union[str, int]]`
        """
        with open(cached_path(file_path), "r") as data_file:   
                        
            for line in self.shard_iterable(data_file.readlines()):
                if not line:
                    continue
                items = json.loads(line)
                text = items[self._text_key]
                # label = items.get(self._label_key)
                labels = items.get(self._label_key)
                if labels is not None:
                    if self._skip_label_indexing:
                        try:
                            # label = int(label)
                            labels = [int(label) for label in labels]
                        except ValueError:
                            raise ValueError(
                                "Labels must be integers if skip_label_indexing is True."
                            )
                    else:
                        # label = str(label)
                        labels = [str(label) for label in labels]
                
                yield self.text_to_instance(text=text, labels=labels)

    @overrides
    def text_to_instance(
        self, text: str, labels: List[Union[str, int]] = None
    ) -> Instance:  # type: ignore
        """
        # Parameters

        text : `str`, required.
            The text to classify
        labels : `List[Union[str, int]]`, optional, (default = `None`).
            The labels for this text.

        # Returns

        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - label (`MultiLabelField`) :
              The labels of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[Field] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens, self._token_indexers))
            fields["tokens"] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields["tokens"] = TextField(tokens, self._token_indexers)
        if labels is not None:
            fields["labels"] = MultiLabelField(
                labels, skip_indexing=self._skip_label_indexing, num_labels=self._num_labels
            )
        return Instance(fields)