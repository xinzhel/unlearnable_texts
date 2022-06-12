from allennlp.modules.token_embedders import Embedding, TokenEmbedder

import logging
from typing import Optional, Tuple, Dict, Any
import torch
from overrides import overrides
from allennlp.data import Vocabulary
logger = logging.getLogger(__name__)


@TokenEmbedder.register("for_pretrained_tokenizer")
class ForPretrainedTokenizerEmbedder(TokenEmbedder):

    def __init__(
        self,
        vocab: Vocabulary = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab=vocab, **kwargs)


    @overrides
    def get_output_dim(self):
        return self.embedding.get_output_dim()

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        return self.embedding(token_ids)
       