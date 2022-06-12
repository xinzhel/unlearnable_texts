import warnings
from typing import Dict, Tuple, Any, cast
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import ROUGE, BLEU
from allennlp.common.lazy import Lazy

from transformers.models.bart.modeling_bart import BartConfig, BartModel, BartForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F
from allennlp_models.generation.models.bart import _BartEncoderWrapper, BartEncoder

DecoderCacheType = Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], ...]



@Model.register("my_bart")
class Bart(Model):
    """ A copy from allennlp_models/generation/bart.py with modifications
    1. allow the model setup with random parameters
    
    """

    def __init__(
        self,
        model_name: str,
        vocab: Vocabulary,
        from_pretrained: bool = True,
        beam_search: Lazy[BeamSearch] = Lazy(BeamSearch),
        indexer: PretrainedTransformerIndexer = None,
        encoder: Seq2SeqEncoder = None,
        **kwargs,
    ):
        super().__init__(vocab)
        
        if from_pretrained:
            self.bart = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            # initialize the model with the config
            model_config = BartConfig.from_pretrained(model_name)
            self.bart = BartForConditionalGeneration(config=model_config)
        self._indexer = indexer or PretrainedTransformerIndexer(model_name, namespace="tokens")

        self._start_id = self.bart.config.bos_token_id  # CLS
        self._decoder_start_id = self.bart.config.decoder_start_token_id or self._start_id
        self._end_id = self.bart.config.eos_token_id  # SEP
        self._pad_id = self.bart.config.pad_token_id  # PAD

        # At prediction time, we'll use a beam search to find the best target sequence.
        # For backwards compatibility, check if beam_size or max_decoding_steps were passed in as
        # kwargs. If so, update the BeamSearch object before constructing and raise a DeprecationWarning
        deprecation_warning = (
            "The parameter {} has been deprecated."
            " Provide this parameter as argument to beam_search instead."
        )
        beam_search_extras = {}
        if "beam_size" in kwargs:
            beam_search_extras["beam_size"] = kwargs["beam_size"]
            warnings.warn(deprecation_warning.format("beam_size"), DeprecationWarning)
        if "max_decoding_steps" in kwargs:
            beam_search_extras["max_steps"] = kwargs["max_decoding_steps"]
            warnings.warn(deprecation_warning.format("max_decoding_steps"), DeprecationWarning)
        self._beam_search = beam_search.construct(
            end_index=self._end_id, vocab=self.vocab, **beam_search_extras
        )

        self._rouge = ROUGE(exclude_indices={self._start_id, self._pad_id, self._end_id})
        self._bleu = BLEU(exclude_indices={self._start_id, self._pad_id, self._end_id})

        # Replace bart encoder with given encoder. We need to extract the two embedding layers so that
        # we can use them in the encoder wrapper
        if encoder is not None:
            assert (
                encoder.get_input_dim() == encoder.get_output_dim() == self.bart.config.hidden_size
            )
            self.bart.model.encoder = _BartEncoderWrapper(
                encoder,
                self.bart.model.encoder.embed_tokens,
                self.bart.model.encoder.embed_positions,
            )

    @overrides
    def forward(
        self, source_tokens: TextFieldTensors, target_tokens: TextFieldTensors = None, metadata = None
    ) -> Dict[str, torch.Tensor]:
        """
        Performs the forward step of Bart.

        # Parameters

        source_tokens : `TextFieldTensors`, required
            The source tokens for the encoder. We assume they are stored under the `tokens` key.
        target_tokens : `TextFieldTensors`, optional (default = `None`)
            The target tokens for the decoder. We assume they are stored under the `tokens` key. If no target
            tokens are given, the source tokens are shifted to the right by 1.


        # Returns

        `Dict[str, torch.Tensor]`
            During training, this dictionary contains the `decoder_logits` of shape `(batch_size,
            max_target_length, target_vocab_size)` and the `loss`. During inference, it contains `predictions`
            of shape `(batch_size, max_decoding_steps)` and `log_probabilities` of shape `(batch_size,)`.

        """
        inputs = source_tokens
        targets = target_tokens
        input_ids, input_mask = inputs["tokens"]["token_ids"], inputs["tokens"]["mask"]

        outputs = {}

        # If no targets are provided, then shift input to right by 1. Bart already does this internally
        # but it does not use them for loss calculation.
        if targets is not None:
            target_ids, target_mask = targets["tokens"]["token_ids"], targets["tokens"]["mask"]
        else:
            target_ids = input_ids[:, 1:]
            target_mask = input_mask[:, 1:]

        if self.training:
            bart_outputs = self.bart(
                input_ids=input_ids,
                attention_mask=input_mask,
                decoder_input_ids=target_ids[:, :-1].contiguous(),
                decoder_attention_mask=target_mask[:, :-1].contiguous(),
                use_cache=False,
                return_dict=True,
            )
            outputs["decoder_logits"] = bart_outputs.logits

            # The BART paper mentions label smoothing of 0.1 for sequence generation tasks
            outputs["loss"] = sequence_cross_entropy_with_logits(
                bart_outputs.logits,
                cast(torch.LongTensor, target_ids[:, 1:].contiguous()),
                cast(torch.BoolTensor, target_mask[:, 1:].contiguous()),
                label_smoothing=0.1,
                average="token",
            )
        else:
            # Use decoder start id and start of sentence to start decoder
            initial_decoder_ids = torch.tensor(
                [[self._decoder_start_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            inital_state = {
                "input_ids": input_ids,
                "input_mask": input_mask,
            }
            beam_result = self._beam_search.search(
                initial_decoder_ids, inital_state, self.take_step
            )

            predictions = beam_result[0]
            max_pred_indices = (
                beam_result[1].argmax(dim=-1).view(-1, 1, 1).expand(-1, -1, predictions.shape[-1])
            )
            predictions = predictions.gather(dim=1, index=max_pred_indices).squeeze(dim=1)

            self._rouge(predictions, target_ids)
            self._bleu(predictions, target_ids)

            outputs["predictions"] = predictions
            outputs["log_probabilities"] = (
                beam_result[1].gather(dim=-1, index=max_pred_indices[..., 0]).squeeze(dim=-1)
            )

            self.make_output_human_readable(outputs)

        return outputs

    @staticmethod
    def _decoder_cache_to_dict(decoder_cache: DecoderCacheType) -> Dict[str, torch.Tensor]:
        cache_dict = {}
        for layer_index, layer_cache in enumerate(decoder_cache):
            # Each layer caches the key and value tensors for its self-attention and cross-attention.
            # Hence the `layer_cache` tuple has 4 elements.
            assert len(layer_cache) == 4
            for tensor_index, tensor in enumerate(layer_cache):
                key = f"decoder_cache_{layer_index}_{tensor_index}"
                cache_dict[key] = tensor
        return cache_dict

    def _dict_to_decoder_cache(self, cache_dict: Dict[str, torch.Tensor]) -> DecoderCacheType:
        decoder_cache = []
        for layer_index in range(len(self.bart.model.decoder.layers)):
            base_key = f"decoder_cache_{layer_index}_"
            layer_cache = (
                cache_dict[base_key + "0"],
                cache_dict[base_key + "1"],
                cache_dict[base_key + "2"],
                cache_dict[base_key + "3"],
            )
            decoder_cache.append(layer_cache)
        assert decoder_cache
        return tuple(decoder_cache)

    def take_step(
        self, last_predictions: torch.Tensor, state: Dict[str, torch.Tensor], step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        # Parameters

        last_predictions : `torch.Tensor`
            The predicted token ids from the previous step. Shape: `(group_size,)`
        state : `Dict[str, torch.Tensor]`
            State required to generate next set of predictions
        step : `int`
            The time step in beam search decoding.


        # Returns

        `Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
            A tuple containing logits for the next tokens of shape `(group_size, target_vocab_size)` and
            an updated state dictionary.
        """
        if len(last_predictions.shape) == 1:
            last_predictions = last_predictions.unsqueeze(-1)

        decoder_cache = None
        decoder_cache_dict = {
            k: state[k].contiguous()
            for k in state
            if k not in {"input_ids", "input_mask", "encoder_states"}
        }
        if len(decoder_cache_dict) != 0:
            decoder_cache = self._dict_to_decoder_cache(decoder_cache_dict)

        encoder_outputs = (state["encoder_states"],) if "encoder_states" in state else None
        outputs = self.bart(
            input_ids=state["input_ids"] if encoder_outputs is None else None,
            attention_mask=state["input_mask"],
            encoder_outputs=encoder_outputs,
            decoder_input_ids=last_predictions,
            past_key_values=decoder_cache,
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]
        log_probabilities = F.log_softmax(logits, dim=-1)

        decoder_cache = outputs.past_key_values
        if decoder_cache is not None:
            decoder_cache_dict = self._decoder_cache_to_dict(decoder_cache)
            state.update(decoder_cache_dict)

        state["encoder_states"] = outputs.encoder_last_hidden_state

        return log_probabilities, state

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """

        # Parameters

        output_dict : `Dict[str, torch.Tensor]`
            A dictionary containing a batch of predictions with key `predictions`. The tensor should have
            shape `(batch_size, max_sequence_length)`

        # Returns

        `Dict[str, Any]`
            Original `output_dict` with an additional `predicted_tokens` key that maps to a list of lists of
            tokens.

        """
        predictions = output_dict["predictions"]
        predicted_tokens = [None] * predictions.shape[0]
        for i in range(predictions.shape[0]):
            predicted_tokens[i] = self._indexer.indices_to_tokens(
                {"token_ids": predictions[i].tolist()},
                self.vocab,
            )
        output_dict["predicted_tokens"] = predicted_tokens  # type: ignore
        output_dict["predicted_text"] = self._indexer._tokenizer.batch_decode(
            predictions.tolist(), skip_special_tokens=True
        )

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics.update(self._rouge.get_metric(reset=reset))
            metrics.update(self._bleu.get_metric(reset=reset))
        return metrics
