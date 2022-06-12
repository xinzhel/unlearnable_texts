import glob
import hashlib
import json
import logging
import os
import random
import re
from copy import Error, deepcopy
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import ftfy
import numpy as np
import torch
from allennlp.common import Lazy, Tqdm, cached_transformers
from allennlp.common import util as common_util
from allennlp.nn.util import move_to_device
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import Batch, DataLoader, Instance, Token
from allennlp.data.data_loaders.data_loader import DataLoader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import IndexField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          SingleIdTokenIndexer,
                                          TokenCharactersIndexer, TokenIndexer)
from allennlp.data.tokenizers import (PretrainedTransformerTokenizer,
                                      SpacyTokenizer, Token, Tokenizer)
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import \
    PretrainedTransformerTokenizer
from allennlp.models.model import Model
from allennlp.modules import token_embedders
from allennlp.modules.token_embedders import token_embedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import \
    PretrainedTransformerEmbedder
from allennlp.nn.util import find_embedding_layer, find_text_field_embedder
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import \
    LearningRateScheduler
from allennlp.training.momentum_schedulers.momentum_scheduler import \
    MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer
from allennlp_models.rc import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import TransformerSquadReader
from allennlp_extra.models.bart import Bart
from allennlp_extra.models.seq2seq import MySeq2Seq

from overrides import overrides
from torch import Tensor, backends
from torch import functional as F
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.utils.hooks import RemovableHandle
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality.part_of_speech import PartOfSpeech

from textattack.constraints import Constraint
from textattack.shared.attacked_text import AttackedText
from textattack.transformations import WordSwap

logger = logging.getLogger(__name__)

"""
Text Modification
============================================
"""
def textfield_to_attackedtext(allennlp_text: TextField) -> AttackedText: 
    words = allennlp_text.human_readable_repr()
    textattack_text = AttackedText(' '.join(words))
    textattack_text._words = words
    return textattack_text

def validate_word_swap(allennlp_text, modification, constraints = [ WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech() ]):
    reference_text = textfield_to_attackedtext(allennlp_text)
    for index, new_word in modification.items():
        transformed_text = reference_text.replace_word_at_index(index, new_word)
        # many textattack constraints only work for `WordSwap` transformation
        transformed_text.attack_attrs["last_transformation"] =  WordSwap()
    for C in constraints:
        if not C(transformed_text, reference_text):
            return False
    return True

def instances_to_tensors(instances, vocab):
    all_batches = Batch(instances)
    all_batches.index_instances(vocab)
    return all_batches.as_tensor_dict()

# self.position_to_modify_constraints = []
#         self.input_field_name = input_field_name
#         if self.input_field_name == "passage":
#             self.position_to_modify_constraints.append(ExcludeAnswerSpan())
# for C in self.position_to_modify_constraints:
#     qualified = C.check(position_to_flip, instance, self.input_field_name)
    

class TextModifier:
    """ 
    Search text modifications in the vocabulary by a gradient x Embeddings way. 
    Specifically, this class aims to:
    1. modify `instances` according to `self.modifications`
    2. generate/update `self.modifications`

    # Parameters

    model: `Model`, required
        used to calculate gradients.
    data_loader: `DatasetReader`, required
        used to load instances from the `data_path`.
    serialization_dir: `str`, required
        used to save or/and load modifications
    """

    def __init__(
        self, 
        model_bundle,
        data_loader: DataLoader,
        serialization_dir: str,
        perturb_bsz: int=32,
        input_field_name: str = "tokens",
        max_swap: int=1,
        class_wise: bool =False,
        only_where_to_modify: bool = False,
        error_max: int = -1, # 1 for maximization, -1 for minimization
        **kwargs, # it is important to add this to avoid errors with extra arguments from command lines for constructing other classes
    ) -> None: 
        # model
        self.model_bundle = model_bundle

        # data
        self.data_loader = data_loader
        self.batch_size = self.data_loader.batch_sampler.batch_size
        self.instances = list(self.data_loader.iter_instances())
        self.num_examples = len(self.instances) # just to ensure _instances is loaded

        # modification
        self.perturb_bsz = perturb_bsz
        self.indices_of_token_to_modify = only_where_to_modify
        self.input_field_name = input_field_name
        self.error_max = error_max
        # initialize modified positions and modifications
        self.max_swap = max_swap
        self.class_wise = class_wise
        if self.class_wise:
            print(self.class_wise)
            assert self.indices_of_token_to_modify is not True
        if not class_wise:
            self.modifications = [{0:'the'} for _ in range(self.num_examples)]
            self.update_positions_to_modify(wir_method='random')
        else:
            self.triggers = {key: ['the'] * self.max_swap for key in self._vocab._token_to_index['labels']} 


    def update(self, epoch, batches_in_epoch_completed):
        print("Update.")
        # update perturbations
        if self.class_wise:
            self.update_triggers(epoch, batches_in_epoch_completed)
        elif self.indices_of_token_to_modify:
            self.update_positions_to_modify()
            self.save_modification(epoch=epoch, batch_idx=batches_in_epoch_completed)
        else:
            self.update_unlearnable()
            # save unlearnable modifications 
            self.save_modification(epoch=epoch, batch_idx=batches_in_epoch_completed, save_text=True)

    def update_unlearnable(self):
        
        all_instances = deepcopy(self.instances)

        # instances to a tensor
        dataset_tensor_dict = instances_to_tensors(all_instances, self.model.vocab)
        grads, _ = get_grad(dataset_tensor_dict, self.model, self.embedding_layer, batch_size=self.batch_size) 

        idx_of_instance = 0
        for idx_of_batch, instance in enumerate(all_instances):
            
            modification_dict = self.generate_modification_by_approx_scores(instance, grads[idx_of_batch], self.model.vocab)
            
            # update modifications
            self.modifications[idx_of_instance] = modification_dict
            idx_of_instance += 1
    
    def generate_modification_by_approx_scores(self, instance, grad, vocab): # (p, s) pairs
        tokens = instance.fields[self.input_field_name].tokens
        seq_len = len(tokens)
        num_vocab = len(vocab._token_to_index[self.input_field_name])
        token_end_idx = self.token_start_idx + (seq_len-1)
        
        # shape: seq_len * num_vocab
        _, indices = \
                get_approximate_scores(grad[self.token_start_idx:token_end_idx, :], self.embedding_matrix, \
                        all_special_ids=self.all_special_ids, sign=self.error_max)
        # update modified postions and modifications
        modifications = [] 
        positions_flipped = [] # check for not flipping the same toke again
        idx_of_modify = 0  
        while len(modifications) < self.max_swap:
            if idx_of_modify >= len(indices): # cannot find any modifications satisfying the constraint
                break
            position_to_flip, what_to_modify = int(indices[idx_of_modify] // num_vocab) + self.token_start_idx, int(indices[idx_of_modify] % num_vocab)
            idx_of_modify += 1
            if position_to_flip not in positions_flipped: # do not modify the same position twice in one iteration
                modify_token = vocab._index_to_token[self.namespace][int(what_to_modify)]

                # validate modification by constraints
                if not validate_word_swap(instance[self.input_field_name], modification={position_to_flip: modify_token})
                    continue
                
                modifications.append((position_to_flip, modify_token))
                positions_flipped.append(position_to_flip)
        
        modification_dict = {}
        for position_to_flip, modify_token in modifications:
            modification_dict[position_to_flip] = modify_token
        return modification_dict


    def update_triggers(self, epoch, batch_idx):
        # always maintain a clean version of `self.instances`
        all_instances = deepcopy(self.instances)
        instances_dict = {key: [] for key in self.model.vocab._token_to_index['labels']}
        for instance in all_instances:
            instances_dict[instance['label'].label].append(instance)

        output = '\n Epoch: ' + str(epoch) + ' || Batch: '+str(batch_idx) + '\n'
        
        for label, instances in instances_dict.items():
            instances = prepend_batch(instances, self.triggers, self.model.vocab, self.input_field_name)
            
            lowest_loss = 9999
            num_no_improvement = 0
            patient = 10
            for batch_indices in self.data_loader.batch_sampler.get_batch_indices(instances):
                batch = []
                for i in batch_indices:
                    batch.append(instances[i])
                # get tensor dict
                batch = Batch(batch)
                batch.index_instances(self.model.vocab)
                dataset_tensor_dict = batch.as_tensor_dict()

                # get gradients
                grads, loss = get_grad(dataset_tensor_dict, self.model, self.embedding_layer) 
                # exist for convergence
                if loss < lowest_loss:
                    lowest_loss = loss
                    num_no_improvement = 0
                else:
                    num_no_improvement += 1
                    if num_no_improvement >= patient:
                        break
                    else:
                        continue
                grads = grads[:,self.max_swap,:]
                grads = np.sum(grads, axis=0)
                first_order_dir = torch.einsum("ij,kj->ik", (torch.Tensor(grads), self.embedding_matrix.to('cpu'))) 
                first_order_dir[:, self.all_special_ids] = np.inf
                new_trigger_ids = (-first_order_dir).argmax(1)
                self.triggers[label] =  [self._vocab.get_token_from_index(int(token_id), namespace=self.namespace) for token_id in new_trigger_ids]
                # print('label||', ' '.join(self.triggers[label]))
            
            output +=  f'   triggers for {label}: ' + ' '.join(self.triggers[label])
        
        return output

    
    def update_positions_to_modify(self, wir_method='gradient'):
        """ Somewhat imitate the implementation from 
        `textattack.GreedyWordSwapWIR._get_index_order()`
        """
        # always maintain a clean version of `self.instances`
        all_instances = deepcopy(self.instances)
        assert len(self.modifications) == len(all_instances)
        if wir_method == "gradient":
            # get tensor dict
            all_batches = Batch(all_instances)
            all_batches.index_instances(self.model.vocab)
            dataset_tensor_dict = all_batches.as_tensor_dict()

            # get gradient
            grads, _ = get_grad(dataset_tensor_dict, self.model, self.embedding_layer)
            for idx_of_instance, instance in enumerate(all_instances):
                
                tokens = instance.fields[self.input_field_name].tokens
                valid_grads = grads[idx_of_instance][:len(tokens),:]
                 # np.sqrt(np.array([g.dot(g) for g in valid_grads]))
                scores_pos_to_modify = np.linalg.norm(valid_grads, axis=1)
                for C in self.position_to_modify_constraints:
                    scores_pos_to_modify = C.apply(scores_pos_to_modify, instance)
                
                index_of_token_to_modify = np.argmax(scores_pos_to_modify)
                self.modifications[idx_of_instance] = {int(index_of_token_to_modify):None}

        elif wir_method == "random":
            for idx_of_instance, instance in enumerate(all_instances):
                qualified = False
                len_tokens = len(instance.fields[self.input_field_name].tokens)
                index_order = np.arange(len_tokens)
                np.random.shuffle(index_order)
                for i in index_order:
                    for C in self.position_to_modify_constraints:
                        qualified = C.check(i, instance, self.input_field_name)
                    if qualified:
                        break
                self.modifications[idx_of_instance] = {int(i): 'the'}
                
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")


    def save_modification(self, epoch=None, batch_idx=None, save_text=False):
        
        if save_text and self.input_field_name == "passage":

            return final_output
            
        else:
            return self.modifications


    def modify(self, batch, batch_indices):
        """ 
        This method is used during training, where 
        `batch_indices`: identify the indices in the same order as `self.instances` and `self.modifications`
        """
        
        # create new batch so the original texts would not be modified
        batch_copy = deepcopy(batch)
        if not self.class_wise:
            for batch_idx, instance in enumerate(batch_copy):
                dataset_idx = batch_indices[batch_idx]
                # change text field of each instance
                for (position_to_modify, substitution) in self.generate_modification(dataset_idx): # `max_swap`
                    self.modify_one_example(instance, position_to_modify, substitution, self.input_field_name)
                instance.index_fields(self._vocab) 
        else:
            batch_copy = prepend_batch(batch_copy, self.triggers, self._vocab, self.input_field_name)
    
        return batch_copy


    @classmethod
    def modify_one_example(
            cls, instance: Instance, \
            position_to_modify:int, \
            substitution: str, \
            input_field_name: str
            ):
            
        instance.fields[input_field_name].tokens[position_to_modify] = Token(substitution )

        if input_field_name == "passage":
            
            # compensate offsets
            passage_length = len(instance.fields["passage"].tokens)
            offsets = instance.fields['metadata'].metadata["token_offsets"]
            orig_start_offset = offsets[position_to_modify][0]
            orig_end_offset = offsets[position_to_modify][1]
            # e.g., substituion is "hello",  modified word is "hi", 
            # compensate_offset would be 5-2=3
            compensate_offset = len(substitution) - (orig_end_offset - orig_start_offset)
            # compensate from the end offset of the modified position
            offsets[position_to_modify] = (offsets[position_to_modify][0], offsets[position_to_modify][1]+compensate_offset)
            for pos in range(position_to_modify+1, passage_length, 1):
                offsets[pos] = (offsets[pos][0] + compensate_offset, offsets[pos][1] + compensate_offset)

            # change orignal text. This is necessary becasue it would be used to calculate F1 metric
            passage_str = instance.fields['metadata'].metadata["original_passage"]
            passage_str = passage_str[:orig_start_offset] + substitution + passage_str[orig_end_offset:]
            # this would not be necessay since they point to the same str object in memory
            instance.fields['metadata'].metadata["original_passage"] = passage_str

            return passage_str

        if input_field_name == 'source_tokens':
            modified_str = ''

            # TODO: 
            # tokenized items -> articles after modifications 
            # (correctly calculate offsets): save_modification, modify_one_example
            # rather than caoncatenate tokens with whitespace
            tokens = instance.fields['source_tokens'].tokens
            for i, token in enumerate(tokens):
                if i == int(position_to_modify):
                    modified_str += str(substitution)
                else:
                    modified_str += str(token)

                if i+1< len(tokens) and str(tokens[i+1]) not in '.,\"\'':
                    modified_str += ' '
 
            return modified_str

    def generate_modification(self, dataset_idx):
        assert self.modifications is not None
        for (position_to_modify, substitution) in self.modifications[dataset_idx].items(): # `max_swap`
            yield int(position_to_modify), substitution


def prepend_batch(instances, trigger_tokens, vocab, input_field_name="tokens"):
    """
    trigger_tokens List[str]
    """
    instances_with_triggers = []
    for instance in deepcopy(instances): 
        instance_perturbed = prepend_instance(instance, trigger_tokens, vocab, input_field_name)
        instances_with_triggers.append(instance_perturbed)
    
    return instances_with_triggers

def prepend_instance(instance, trigger_tokens: List[Token], vocab=None, input_field_name="tokens", position = 'begin'):
    instance_copy = deepcopy(instance)
    label = instance_copy.fields['label'].label
    if 'premise' in instance_copy.fields: # NLI
        assert vocab is not None
        # TODO: inputs for transformers
        instance_copy.fields['hypothesis'].tokens = trigger_tokens[label] + instance_copy.fields['hypothesis'].tokens
        instance_copy.fields['hypothesis'].index(vocab) 

    else:# text classification
        if str(instance_copy.fields[input_field_name].tokens[0]) == '[CLS]':

            if position == 'begin':
                instance_copy.fields[input_field_name].tokens = [instance_copy.fields[input_field_name].tokens[0]] + \
                    trigger_tokens[label] + instance_copy.fields[input_field_name].tokens[1:]
            elif position == 'end':
                instance_copy.fields[input_field_name].tokens = instance_copy.fields[input_field_name].tokens + trigger_tokens[label]
            elif position == 'middle':
                seq_len = len(instance_copy.fields[input_field_name].tokens)
                insert_point = seq_len // 2
                instance_copy.fields[input_field_name].tokens = instance_copy.fields[input_field_name].tokens[:insert_point] + \
                    trigger_tokens[label] + instance_copy.fields[input_field_name].tokens[insert_point:]
        else:
            if position == 'begin':
                instance_copy.fields[input_field_name].tokens =  trigger_tokens[label] + instance_copy.fields[input_field_name].tokens
            elif position == 'end':
                instance_copy.fields[input_field_name].tokens = instance_copy.fields[input_field_name].tokens + trigger_tokens[label]
            elif position == 'middle':
                seq_len = len(instance_copy.fields['tokens'].tokens)
                insert_point = seq_len // 2
                instance_copy.fields[input_field_name].tokens = instance_copy.fields[input_field_name].tokens[:insert_point] + \
                    trigger_tokens[label] + instance_copy.fields[input_field_name].tokens[insert_point:]
        
        if vocab is not None:
            instance_copy.fields[input_field_name].index(vocab)
    
    return instance_copy

"""
search scores
============================================
"""
def get_approximate_scores(grad, embedding_matrix, all_special_ids: List[int] =[], sign: int = -1):
    """ The objective is to minimize the first-order approximate of L_adv:
        L = L(orig_text) + [replace_token - orig_text[i]].dot(input_grad)
        ignore `orig_text` since it is constant and does not affect the result:
        minimize: replace_token.dot(input_grad)

        grad: (seq_len, embed_size); we assume all positions in the first dimension are 
            valid, i.e.,no special positions like [SEP] [PAD]
        embedding_matrix: (vocab_size, embed_size)
        all_special_ids: block invalid tokens in vocabulary (or embedding matrix)
        sign: -1 for minimization ; 1 for maximization
    """
    
    # (seq_len, vocab_size)
    first_order_dir = torch.einsum("ij,kj->ik", (torch.Tensor(grad), embedding_matrix.to('cpu'))) 

    # TODO: score in the replacement dimension for constraints or ...
    # use MLM to generate probability and then weight the above score

    # block invalid replacement
    first_order_dir[:, all_special_ids] = (-sign)*np.inf # special tokens are invalid for replacement

    scores = first_order_dir.flatten()
    if sign == -1: 
        descend = False # small score first: low -> high
    else:
        descend = True # large score first
    # get scores for the replacements for all position (seq_len*num_vocab)
    scores, indices = scores.sort(dim=-1, descending=descend)
    
    return scores, indices 

"""
Gradients
============================================
"""
def get_grad(
    dataset_tensor_dict, 
    model: torch.nn.Module, 
    layer: torch.nn.Module, 
    loss_fct: _Loss = None,
    batch_size: int = 16):
    """ 
    # Parameters

    batch: A dictionary containing model input
    model: (1) the subclass of the `PreTrainedModel`  or 
           (2) Pytorch model with a method "get_input_embeddings" which return `nn.Embeddings`
    layer: the layer of `model` to get gradients, e.g., a embedding layer
    batch_size: avoid the case that `instances` may be too overloaded to perform forward/backward pass

    # Return

    return_grad: shape (batch size, sequence_length, embedding_size): gradients for all tokenized elements
        , including the special prefix/suffix and <SEP>.
    """
    
    cuda_device = next(model.parameters()).device

    gradients: List[Tensor] = []
    # register hook
    hooks: List[RemovableHandle] = _register_gradient_hooks(gradients, layer)

    # require grads for all model params 
    original_param_name_to_requires_grad_dict = {}
    for param_name, param in model.named_parameters():
        original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
        param.requires_grad = True

    # calculate grad for inference network
    orig_mode = model.training
    model.train(mode=False)
 
    # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
    with backends.cudnn.flags(enabled=False):
        
        dataset_tensor_dict = move_to_device(dataset_tensor_dict, cuda_device)

        # update in batch 
        gradients_for_all = []
        total_loss = 0.0
        # Workaround to batch tensor_dict rather than instances 
        # in order to return a gradients list with the same sequence length
        # to be concatenated
        dataset_tensor_dict_iterator = loop_dict(dataset_tensor_dict, function_on_val=lambda val : iter(val))
        for batch in batch_dataset_tensor_dict_generator(dataset_tensor_dict_iterator,  batch_size=batch_size):
            batch.pop('metadata', None)
            outputs = model.forward(**batch)  

            if loss_fct is None:
                loss = outputs["loss"]
            else:
                raise NotImplementedError("Not support the customized loss function.")
                # labels = batch['labels'].view(-1)
                # loss = loss_fct(outputs['logits'], labels)
            # Zero gradients.
            # NOTE: this is actually more efficient than calling `model.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for p in model.parameters():
                p.grad = None
            gradients.clear()
            loss.backward()
            total_loss += loss.detach().cpu()
            if isinstance(model, Bart):
                gradients_for_all.append(gradients[1].detach().cpu().numpy())
            elif isinstance(model, MySeq2Seq ):
                gradients_for_all.append(gradients[-1].detach().cpu().numpy())
            else:
                gradients_for_all.append(gradients[0].detach().cpu().numpy())

    if len(gradients) != 1:
        import warnings
        warnings.warn(
            """get_grad: gradients for >1 inputs are acquired. 
            This should still work well for bidaf and 
            since the 1-st tensor is for passage.""")

    for hook in hooks:
        hook.remove()

    # restore the original requires_grad values of the parameters
    for param_name, param in model.named_parameters():
        param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
    model.train(mode=orig_mode)
    
    return_grad = np.concatenate(gradients_for_all, axis=0)
       
    return return_grad, total_loss

def loop_dict(input_dict, function_on_val):
    result_dict = {}
    for name, val in input_dict.items():
        if isinstance(val, dict): 
            result_dict[name] = loop_dict(val, function_on_val)
        elif isinstance(val, Iterable):
            result_dict[name] = function_on_val(val)
        else:
            raise Error("The Value has an Unknown Types.")
    return result_dict

def batch_dataset_tensor_dict_generator(dataset_tensor_dict_iterator, batch_size=16):
    ensure_iterable_value = False
    for _, value in dataset_tensor_dict_iterator.items():
        if isinstance(value, Iterable):
            ensure_iterable_value = True
    if not ensure_iterable_value:
        raise Exception('Have to ensure iterable value.')
    
    def func(iterator):
        lst = list(islice(iterator, batch_size))
        
        if len(lst) > 0 and isinstance(lst[0], torch.Tensor):
            return torch.stack(lst)
        else:
            return lst
    def runout(d):
        for _, value in d.items():
            if isinstance(value, torch.Tensor) or isinstance(value, list):
                if len(value) == 0:
                    return True
                else: 
                    return False
            elif isinstance(value, dict):
                return runout(value)
            else:
                raise Exception()

    while True:
        s = loop_dict(dataset_tensor_dict_iterator, function_on_val=func)
        if runout(s):
            break
        else:
            yield s

def _register_gradient_hooks(gradients, layer):

    def hook_layers(module, grad_in, grad_out):
        grads = grad_out[0]
        gradients.append(grads)

    hooks = []
    hooks.append(layer.register_full_backward_hook(hook_layers))
    return hooks

"""
DataReader Wrappers for Perturbing Original AllenNLP DataReaders
================================================================
"""
@DatasetReader.register("perturb_labeled_text")
class PerturbLabeledTextDatasetReader(DatasetReader):
    def __init__(
            self, 
            dataset_reader: DatasetReader,
            perturb_split: str = 'train',
            modification_path: str = None,
            fix_substitution: str = None,
            random_postion: bool = False,
            triggers: dict = None, # fix_insertion
            position: str = 'begin',
            perturb_prob: float = 1.0,
            max_perturbed_instances: int = None,
            skip: bool = False,
            **kwargs,):
        super().__init__(**kwargs)
        self._reader = dataset_reader
        self.perturb_split = perturb_split
        self.modifications = json.load(open(modification_path, 'rb')) if modification_path else None
        
        if triggers is not None and isinstance(triggers, dict): # dict for classification, list for rc
            if isinstance( self._reader._tokenizer, PretrainedTransformerTokenizer):
                for label, trigger_txt in triggers.items():
                    tokens = self._reader._tokenizer.tokenize(' '.join(trigger_txt))
                    triggers[label] = tokens[1:-1] # exclude [CLS] , [SEP]
                
            else:
                for label, trigger_txt in triggers.items():
                    triggers[label] = [Token(t) for t in trigger_txt]
        self.triggers = triggers
        self.perturb_prob = perturb_prob
        if self.perturb_prob != 1 and max_perturbed_instances is not None:
            raise Exception("`perturb_prob` and `max_perturbed_instances` are mutually exclusive.")
        self.max_perturbed_instances = max_perturbed_instances
        self.skip = skip
        self.position = position
      
        self.fix_substitution = fix_substitution
        self.random_position = random_postion
        

    @overrides
    def _read(self, file_path):
        perturb_idx = 0
        random.seed(13370)
        # perturb_indices = list(range(0, 3200)) + list(range(4800, 5600)) # 0-99 steps, 149- 174 steps

        for instance in self._reader._read(file_path):
            
            if self.perturb_split in file_path and \
                (self.max_perturbed_instances is None or \
                    perturb_idx < self.max_perturbed_instances) and \
                random.uniform(0,1) <= self.perturb_prob : #and (perturb_idx in perturb_indices): 
                logger.info(f'perurb {perturb_idx}')

                if self.skip:
                    continue
                if "passage" in instance.fields:
                    # not affect the original passage since it would be used for other QA pairs
                    instance.fields['passage'] = deepcopy(instance.fields['passage'])
                    instance.fields['metadata'].metadata["original_passage"] = deepcopy(instance.fields['metadata'].metadata["original_passage"])
                    if self.modifications is not None: 
                        instance = self.perturb_squad(instance,)
                    elif self.triggers is not None: 
                        
                        instance = insert_trigger(instance, self.triggers[0], position=self.position)
                    
                else:
                    instance = self.perturb_labeled_single_sent(instance, perturb_idx)
            #         instance.add_field("unlearnable", FlagField(True))
            # else:
            #     instance.add_field("unlearnable", FlagField(False))

            perturb_idx += 1
            yield instance

    def apply_token_indexers(self, instance: Instance) -> None:
        self._reader.apply_token_indexers(instance)

    def perturb_labeled_single_sent(self, instance, perturb_idx):
        label = instance.fields['label'].label
        tokens = instance.fields['tokens'].tokens
        
        if self.modifications is not None:  
            logging.warning(" Tokenizer may differ from the one generating modificaitons. ")
            modification =  [(k, v) for k, v in self.modifications[perturb_idx].items()]
            where_to_modify, what_to_modify = modification[0]
            tokens[int(where_to_modify)] = what_to_modify
            instance = self._reader.text_to_instance(tokens, label)
        elif self.triggers is not None: 
            instance = prepend_instance(instance, self.triggers, position=self.position)
        else:
            logger.warning('You do not specify the perturbation mode (sample-wise | class-wise | skip)')
            # this is used to ensure each batch has examples same as perturbing all examples
            
        return instance
        

    def perturb_squad(self, instance):
        
        id = instance.fields['metadata'].metadata['id']
        assert id is not None
        modification = self.modifications[id]
        position_to_modify, substitution = modification['modified_position'], modification['substitution_word']
        
        
        if self.fix_substitution is not None:
            substitution = self.fix_substitution

        if self.random_position: # this constructs a baseline for evaluating the positions
            len_tokens = len(instance.fields['passage'].tokens)
            index_order = np.arange(len_tokens)
            np.random.shuffle(index_order)
            for i in index_order:
                if ExcludeAnswerSpan().check(i, instance):
                    position_to_modify = i
                break
        position_to_modify = int(position_to_modify)
        instance.fields['passage'].tokens[position_to_modify] = Token(substitution)
        
        # deal with original_str and offsets
        _ = TextModifier.modify_one_example(instance, position_to_modify, substitution, "passage")
        
        return instance

def insert_trigger(instance: Instance, trigger_token: str, position: str="begin"):
    """ add triggers in the beginning/middle/end of the answer spans
    """
    # change passage, answer start/end fields
    insert_token = Token(trigger_token.strip())
    orig_tokens = instance.fields['passage'].tokens
    answer_start = instance.fields['span_start'].sequence_index
    answer_end = instance.fields['span_end'].sequence_index
    if position == 'begin':
        instance.fields['passage'].tokens = orig_tokens[:answer_start] + [insert_token] + orig_tokens[answer_start:]
        instance.fields['span_start'].sequence_index = answer_start + 1
        instance.fields['span_end'].sequence_index = answer_end + 1

        
        # metardata field: deal with original_str and offsets
        # change offsets for insert token
        insert_str = trigger_token.strip() + ' '
        offsets = instance.fields['metadata'].metadata["token_offsets"]
        orig_start_offset = offsets[answer_start][0]
        compensate_offset = len(insert_str)
        offset_for_insert_token = (offsets[answer_start][0], offsets[answer_start][0]+compensate_offset)
        offsets.insert(answer_start, offset_for_insert_token) 
        
        passage_length = len(instance.fields['passage'].tokens)
        # compensate from the end offset of the modified position
        for pos in range(answer_start+1, passage_length, 1):
            offsets[pos] = (offsets[pos][0] + compensate_offset, offsets[pos][1] + compensate_offset)
        instance.fields['metadata'].metadata["token_offsets"] = offsets

        # change orignal text. This is necessary becasue it would be used to calculate F1 metric
        passage_str = instance.fields['metadata'].metadata["original_passage"]
        passage_str = passage_str[:orig_start_offset] + insert_str + passage_str[orig_start_offset:]
        # this is necessay since new str change memory location
        instance.fields['metadata'].metadata["original_passage"] = passage_str
        
        return instance
    else:
        raise Exception(f'{position} is Not Supported Inserted Position.')
    
@DatasetReader.register("perturbed_transformer_squad")
class PerturbedTransformerSquadReader(TransformerSquadReader):
    def __init__(
        self, 
        modification_path: str = None,
        triggers: List[str] = None,
        max_perturbed_instances: int = None,
        transformer_model_name: str = "bert-base-cased", 
        length_limit: int = 384, stride: int = 128, 
        skip_impossible_questions: bool = False, 
        max_query_length: int = 64, 
        tokenizer_kwargs: Dict[str, Any] = None, 
        **kwargs) -> None:
        super().__init__(
            transformer_model_name=transformer_model_name, 
            length_limit=length_limit, 
            stride=stride, 
            skip_impossible_questions=skip_impossible_questions, 
            max_query_length=max_query_length, tokenizer_kwargs=tokenizer_kwargs, 
            **kwargs
            )
        
        self.modifications = json.load(open(modification_path, 'rb')) if modification_path else None
        self.max_perturbed_instances = max_perturbed_instances 
        self.triggers = triggers
    
    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json["data"]
        logger.info("Reading the dataset")
        yielded_question_count = 0
        questions_with_more_than_one_instance = 0
        perturb_id = 0
        for article in dataset:
            for paragraph_json in article["paragraphs"]:
                # perturb
                cache_context = paragraph_json["context"]
                tokenized_context = self._tokenize_context(cache_context)

                for question_answer in self.shard_iterable(paragraph_json["qas"]):
                    first_answer_offset = None
                    answers = [answer_json["text"] for answer_json in question_answer["answers"]]
                    if "train" in file_path and (self.max_perturbed_instances is None or perturb_id < self.max_perturbed_instances):
                        if self.modifications is not None:
                            id = question_answer.get("id", None)
                            assert id is not None
                            context = self.modifications[id]['modified_passage']
                            
                            modification = self.modifications[id]
                            logger.info("Modify:" + str(perturb_id) + ' with substitution '+ str(modification['substitution_word']) + ' at distance_to_answer '+str(modification['distance_to_answer']) )
                            if len(answers) > 0 :
                                first_answer_offset = int(question_answer["answers"][0]["answer_start"]) + (len(modification['substitution_word'])-len(modification['modified_word']))
                        elif self.triggers is not None:
                            trigger_token = self.triggers[0] + ' '
                            first_answer_offset = int(question_answer["answers"][0]["answer_start"])
                            context = paragraph_json["context"][:first_answer_offset] + trigger_token + paragraph_json["context"][first_answer_offset:]
                            first_answer_offset = first_answer_offset + len(trigger_token)
                        else:
                            raise Exception('Not specify triggers or modification path.')
                        cached_tokenized_context = self._tokenize_context(context)
                        perturb_id += 1
                    else:
                        context = cache_context
                        cached_tokenized_context = tokenized_context
    
                        # Just like huggingface, we only use the first answer for training.
                        if len(answers) > 0:
                            first_answer_offset = int(question_answer["answers"][0]["answer_start"])
                        else:
                            first_answer_offset = None
                    
                    instances = self.make_instances(
                        question_answer.get("id", None),
                        question_answer["question"],
                        answers,
                        context,
                        first_answer_offset=first_answer_offset,
                        always_add_answer_span=True,
                        is_training=True,
                        cached_tokenized_context=cached_tokenized_context,
                    )
                    instances_yielded = 0
                    for instance in instances:
                        yield instance
                        instances_yielded += 1
                    if instances_yielded > 1:
                        questions_with_more_than_one_instance += 1
                    yielded_question_count += 1

        if questions_with_more_than_one_instance > 0:
            logger.info(
                "%d (%.2f%%) questions have more than one instance",
                questions_with_more_than_one_instance,
                100 * questions_with_more_than_one_instance / yielded_question_count,
            )

#     comment for `make_instances()` from the line `stride_start = 0` to the end
#     When context+question are too long for the length limit, we emit multiple instances for one question,
#     where the context is shifted. The parameter, stride (default=`128`), specifies the overlap between the shifted context window. It
#     is called "stride" instead of "overlap" because that's what it's called in the original huggingface
#     implementation.
        
# class PerturbedCNNDailyMailDatasetReader(CNNDailyMailDatasetReader):
#     
#     def __init__(
#             self, 
#             modification_path: str = None,
#             source_tokenizer: Tokenizer = None,
#             target_tokenizer: Tokenizer = None,
#             source_token_indexers: Dict[str, TokenIndexer] = None,
#             target_token_indexers: Dict[str, TokenIndexer] = None,
#             source_max_tokens: Optional[int] = None,
#             target_max_tokens: Optional[int] = None,
#             source_prefix: Optional[str] = None,
#             **kwargs) -> None:
#             super().__init__(
#                 source_tokenizer,
#                 target_tokenizer,
#                 source_token_indexers,
#                 target_token_indexers,
#                 source_max_tokens,
#                 target_max_tokens,
#                 source_prefix,
#                 **kwargs,
#                 )
#             self.modifications = json.load(open(modification_path, 'rb')) if modification_path else None
# 
#     @overrides
#     def _read(self, file_path: str):
#         # Reset exceeded counts
#         self._source_max_exceeded = 0
#         self._target_max_exceeded = 0
# 
#         url_file_path = cached_path(file_path, extract_archive=True)
#         data_dir = os.path.join(os.path.dirname(url_file_path), "..")
#         cnn_stories_path = os.path.join(data_dir, "cnn_stories")
#         dm_stories_path = os.path.join(data_dir, "dm_stories")
# 
#         cnn_stories = {Path(s).stem for s in glob.glob(os.path.join(cnn_stories_path, "*.story"))}
#         dm_stories = {Path(s).stem for s in glob.glob(os.path.join(dm_stories_path, "*.story"))}
# 
#         with open(url_file_path, "r") as url_file:
#             for url in url_file:
#                 url = url.strip()
# 
#                 url_hash = self._hashhex(url.encode("utf-8"))
# 
#                 if url_hash in cnn_stories:
#                     story_base_path = cnn_stories_path
#                 elif url_hash in dm_stories:
#                     story_base_path = dm_stories_path
#                 else:
#                     raise ConfigurationError(
#                         "Story with url '%s' and hash '%s' not found" % (url, url_hash)
#                     )
# 
#                 story_path = os.path.join(story_base_path, url_hash) + ".story"
#                 article, summary = self._read_story(story_path)
#                 if url_hash not in self.modifications:
#                     continue
#                 article = self.modifications[url_hash]['modified_article']
#                 if len(article) == 0 or len(summary) == 0 or len(article) < len(summary):
#                     continue
# 
#                 instance = self.text_to_instance(url_hash, article, summary, ) 
#                 if instance is not None: # save url_hash as uniqueid for each instance 
#                     yield instance
