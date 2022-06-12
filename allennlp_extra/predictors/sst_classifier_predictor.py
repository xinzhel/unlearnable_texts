from overrides import overrides
from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
import torch




from torch import backends
from allennlp.data.batch import Batch
from allennlp.nn import util
from overrides import overrides

@Predictor.register('sst_classifier')
class SstClassifierPredictor(Predictor):
    """"Predictor wrapper for the SstClassifier"""
    
    @overrides
    def _json_to_instance(self, json_dict):
        from allennlp.data import Instance
        from copy import deepcopy
        if isinstance(json_dict, dict):
            fields = deepcopy(json_dict) # ensure returned `fields` of `Instance` has different memory id from `json_dict`
            return Instance(fields) 
        elif isinstance(json_dict, Instance):
            return json_dict
        else:
            raise TypeError("Input has to be dictionary or Instance object")

    def predictions_to_labeled_instances(self, instance, output_dict):
        from allennlp.data.fields import LabelField
        label_id = int(output_dict['logits'].argmax()) # this could be used as _label_id for `LabelField`
        instance.add_field('label', LabelField(1, skip_indexing=True) )
        return [instance]

    def get_interpretable_layer(self) -> torch.nn.Module:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        from transformers.models.bert.modeling_bert import BertEmbeddings
        from transformers.models.albert.modeling_albert import AlbertEmbeddings
        from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
        from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
        from allennlp.modules.text_field_embedders.basic_text_field_embedder import (
            BasicTextFieldEmbedder,
        )
        from allennlp.modules.token_embedders.embedding import Embedding
        from transformers.models.distilbert import modeling_distilbert

        for module in self._model.modules():
            if isinstance(module, BertEmbeddings):
                return module.word_embeddings
            if isinstance(module, RobertaEmbeddings):
                return module.word_embeddings
            if isinstance(module, AlbertEmbeddings):
                return module.word_embeddings
            if isinstance(module, GPT2Model):
                return module.wte
            if isinstance(module, modeling_distilbert.Embeddings):
                return module.word_embeddings

        for module in self._model.modules():
            if isinstance(module, TextFieldEmbedder):

                if isinstance(module, BasicTextFieldEmbedder):
                    # We'll have a check for single Embedding cases, because we can be more efficient
                    # in cases like this.  If this check fails, then for something like hotflip we need
                    # to actually run the text field embedder and construct a vector for each token.
                    if len(module._token_embedders) == 1:
                        embedder = list(module._token_embedders.values())[0]
                        if isinstance(embedder, Embedding):
                            if embedder._projection is None:
                                # If there's a projection inside the Embedding, then we need to return
                                # the whole TextFieldEmbedder, because there's more computation that
                                # needs to be done than just multiply by an embedding matrix.
                                return embedder
                return module
        raise RuntimeError("No embedding module found!")

    
    @overrides
    def get_gradients(self, instances: List[Instance]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Gets the gradients of the loss with respect to the model inputs.

        # Parameters

        instances : `List[Instance]`

        # Returns

        `Tuple[Dict[str, Any], Dict[str, Any]]`
            The first item is a Dict of gradient entries for each input.
            The keys have the form  `{grad_input_1: ..., grad_input_2: ... }`
            up to the number of inputs given. The second item is the model's output.

        # Notes

        Takes a `JsonDict` representing the inputs of the model and converts
        them to [`Instances`](../data/instance.md)), sends these through
        the model [`forward`](../models/model.md#forward) function after registering hooks on the embedding
        layer of the model. Calls `backward` on the loss and then removes the
        hooks.
        """
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self._model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True

        embedding_gradients: List[Tensor] = []
        hooks: List[RemovableHandle] = self._register_embedding_gradient_hooks(embedding_gradients)

        for instance in instances:
            self._dataset_reader.apply_token_indexers(instance)

        dataset = Batch(instances)
        dataset.index_instances(self._model.vocab)
        dataset_tensor_dict = util.move_to_device(dataset.as_tensor_dict(), self.cuda_device)
        # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
        with backends.cudnn.flags(enabled=False):
            outputs = self._model.make_output_human_readable(
                self._model.forward(**dataset_tensor_dict)  # type: ignore
            )

            loss = outputs["loss"]
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self._model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in self._model.parameters():
                p.grad = None
            loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # restore the original requires_grad values of the parameters
        for param_name, param in self._model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        return grad_dict, outputs
