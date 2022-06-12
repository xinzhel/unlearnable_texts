import warnings
from typing import Dict, Tuple, Any, cast


from allennlp.data import Vocabulary
from allennlp.data.fields.index_field import IndexField
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU
from allennlp.common.lazy import Lazy

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

DecoderCacheType = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]


@Model.register("gpt2")
class GPT2(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        indexer: PretrainedTransformerIndexer,
        model_name: str = 'gpt2',
        **kwargs,
    ):
        super().__init__(vocab)
        self.gpt = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt.resize_token_embeddings(len(indexer._tokenizer))
        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.gpt.config.bos_token_id  # CLS
        self._end_id = self.gpt.config.eos_token_id  # SEP
        self._pad_id = self._indexer._tokenizer.pad_token_id  # PAD

        # metrics
        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})

        self.loss_fct = CrossEntropyLoss(ignore_index=self._pad_id) #ignores padding token for loss calculation


    def forward(
        self, source_tokens: TextFieldTensors, sep_idx: IndexField, **kwargs,
    ) -> Dict[str, torch.Tensor]:

        inputs = source_tokens
        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        outputs = {}

        if self.training:
            gpt_outputs = self.gpt(
                input_ids=input_ids,
                # attention_mask=input_mask, # why we donot need provide `attention_mask`
            )
            outputs["decoder_logits"] = gpt_outputs[0]
            
            loss = 0
            for i in range(input_ids.size(0)):
                # shift_logits[0] is to predict the next token after sep
                shift_logits = outputs["decoder_logits"][i, sep_idx:-1, :].contiguous() 
                # target_ids[0] is the 1st token after sep
                target_ids = input_ids[i, sep_idx+1:].contiguous()
                loss += self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
            outputs["loss"] = loss
        else:
             
            for i in range(input_ids.size(0)):
                # target_ids[0] is the 1st token after sep
                target_ids = input_ids[i, sep_idx+1:].contiguous()
                # predictions[0, 0] is for the 1st token after sep
                predictions = self.gpt.generate(
                    input_ids=input_ids[i, :sep_idx+1].unsqueeze(0),
                    # max_new_tokens=len(target_ids), # doesnot work as expected
                    min_length=input_ids[i,...].size(0),
                    max_length=input_ids[i,...].size(0),
                    temperature=1.0,
                    top_k=0,
                    top_p=0.9,
                    num_beams=1,
                    repetition_penalty=1,
                    do_sample=True,
                    num_return_sequences=1,
                )
                
                self._rouge(predictions[...,-len(target_ids):], target_ids.unsqueeze(0))
                self._bleu(predictions[...,-len(target_ids):], target_ids.unsqueeze(0))

        return outputs

    

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics.update(self._rouge.get_metric(reset=reset))
            metrics.update(self._bleu.get_metric(reset=reset))
        return metrics

    default_predictor = "seq2seq"