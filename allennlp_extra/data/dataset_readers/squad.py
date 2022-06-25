import json
import logging
from typing import Any, Dict, List, Tuple, Optional

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer

from allennlp_models.rc.dataset_readers import SquadReader
logger = logging.getLogger(__name__)

@DatasetReader.register("du_squad")
class SquadReader(SquadReader):

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__( **kwargs)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
            if type(dataset) == dict:
                dataset = dataset['data']
        logger.info("Reading the dataset")
        idx = 0 # for debug
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in self.shard_iterable(paragraph_json["qas"]):
                    question_text = question_answer["question"].strip().replace("\n", "")
                    is_impossible = question_answer.get("is_impossible", False)
                    if is_impossible:
                        answer_texts: List[str] = []
                        span_starts: List[int] = []
                        span_ends: List[int] = []
                    else:
                        answer_texts = [answer["text"] for answer in question_answer["answers"]]
                        span_starts = [
                            answer["answer_start"] for answer in question_answer["answers"]
                        ]
                        span_ends = [
                            start + len(answer) for start, answer in zip(span_starts, answer_texts)
                        ]
                    additional_metadata = {"id": question_answer.get("id", None)}
                    instance = self.text_to_instance(
                        question_text,
                        paragraph,
                        is_impossible=is_impossible,
                        char_spans=zip(span_starts, span_ends),
                        answer_texts=answer_texts,
                        passage_tokens=tokenized_paragraph,
                        additional_metadata=additional_metadata,
                    )
                    if instance is not None:
                        idx += 1
                        if idx == 999:
                            print('hello')
                            pass # for debug
                        yield instance
