import json
import logging
import os
import random
import pickle
from copy import Error, deepcopy
from itertools import islice
from typing import Any, Dict, Iterable, List, Type, Union

import ftfy
import numpy as np
import torch
from allennlp.common import (Lazy, Params, Registrable, Tqdm,
                             cached_transformers)
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data import (Batch, DataLoader, DatasetReader, Instance, Token,
                           Vocabulary)
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
from allennlp.nn.util import (find_embedding_layer, find_text_field_embedder,
                              move_to_device)
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import \
    LearningRateScheduler
from allennlp.training.momentum_schedulers.momentum_scheduler import \
    MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.trainer import Trainer
from allennlp_models.rc import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import TransformerSquadReader
from overrides import overrides
from textattack.constraints import Constraint
from textattack.constraints.grammaticality.part_of_speech import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import \
    UniversalSentenceEncoder
from textattack.shared.attacked_text import AttackedText
from textattack.transformations import WordSwap
from torch import Tensor, backends
from torch import functional as F
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.utils.hooks import RemovableHandle

from allennlp_extra.models.bart import Bart
from allennlp_extra.models.seq2seq import MySeq2Seq

logger = logging.getLogger(__name__)

"""
Construct Model and/or DataLoader
============================================
"""
class GenerateUnlearnable(Registrable):

    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
        self,
        model,
        vocab,
        data_loader,
        test_data_loader,
    ):
        self.model = model 
        self.data_loader = data_loader
        self.vocab = vocab
        self.test_data_loader = test_data_loader

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        dataset_reader: DatasetReader,
        train_data_path: Any,
        data_loader: Lazy[DataLoader],
        validation_dataset_reader: DatasetReader=None,
        test_data_path: Any=None,
        validation_data_loader: Lazy[DataLoader]=None,
        model: Lazy[Model]=None,  
    ):

        # vocab
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if not os.path.exists(vocab_dir):
           vocab_dir = os.path.join(serialization_dir, "vocabulary.tar.gz")
        vocabulary = Vocabulary.from_files(directory=vocab_dir)
        # data loader
        data_loader = data_loader.construct(reader=dataset_reader, data_path=train_data_path)
        data_loader.index_with(vocabulary)

        # model
        if model is not None:
            model = model.construct(vocab=vocabulary)

        test_data_loader = None
        if validation_data_loader and test_data_path and validation_dataset_reader:
            test_data_loader = validation_data_loader.construct(reader=validation_dataset_reader, data_path=test_data_path)
            test_data_loader.index_with(vocabulary)

        return cls( model=model, vocab=vocabulary, data_loader=data_loader, test_data_loader=test_data_loader )     

GenerateUnlearnable.register("default", constructor="from_partial_objects")(GenerateUnlearnable)



"""
Generate Modification for a Batch/Dataset of instances
======================================================
"""
def instances_to_tensors(instances, vocab):
    all_batches = Batch(instances)
    all_batches.index_instances(vocab)
    return all_batches.as_tensor_dict()

class ExcludeAnswerSpan():
    def __init__(self) -> None:
        pass
    @classmethod
    def apply(cls, scores_pos_to_modify, instance):
        # add constraints for positions to modify: not modify words in correct span
        span_start, span_end = instance.fields['span_start'].sequence_index, instance.fields['span_end'].sequence_index
        scores_pos_to_modify[span_start: span_end+1] = [-float('inf')] * (span_end+1-span_start)
        return scores_pos_to_modify

    @classmethod
    def check_valid_positions(cls, pos_to_modify, instance):
        span_start, span_end = instance.fields['span_start'].sequence_index, instance.fields['span_end'].sequence_index
        return pos_to_modify < span_start or pos_to_modify > span_end

    @classmethod
    def generate_invalid_positions(cls, instance):
        span_start, span_end = instance.fields['span_start'].sequence_index, instance.fields['span_end'].sequence_index
        return list(range(span_start, span_end+1))
from tqdm import tqdm
def generate_modifications(instances, model_bundle, mod_generator, mod_applicator, current_modifications=None, invalid_positions=[], field_to_modify="tokens", vocab_namespace="tokens", batch_size=32, error_max=-1):
    
    if current_modifications is not None:
        mod_applicator.apply_modifications(instances, current_modifications)

    # gradients
    dataset_tensor_dict = instances_to_tensors(instances, model_bundle.vocab)
    gradients, _ = get_grad(dataset_tensor_dict, model_bundle.model, model_bundle.embedding_layer, batch_size=batch_size) 

    # words for all the instances
    all_words = list()
    for instance in instances:
        text_field = instance[field_to_modify]
        all_words.append(text_field.human_readable_repr())
    
    # for Squad
    if field_to_modify == "passage":
        invalid_positions_lst = [ExcludeAnswerSpan.generate_invalid_positions(instance) for instance in instances]
    else:
        invalid_positions_lst = [list() for _ in range(len(instances))]

    # initialize modifications
    # batch_size = self.data_loader.batch_sampler.batch_size
    modifications = [{-1: None} for _ in range(len(instances))] 
    for i, (words, grad, invalid_positions) in enumerate(tqdm(zip(all_words, gradients, invalid_positions_lst))):
        
        modification_dict = mod_generator.generate_modifications(
            words = words,
            grad = grad, 
            index_to_token=model_bundle.vocab._index_to_token[vocab_namespace],
            invalid_position=invalid_positions,
            error_max=error_max
        )
        
        modifications[i] = modification_dict
    return modifications

"""
Modificatoin Generator
============================================
"""
def words_to_attackedtext(words: List[str]) -> AttackedText: 
    textattack_text = AttackedText(' '.join(words))
    textattack_text._words = words
    return textattack_text

def validate_word_swap(words, modification, constraints = [ WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech() ]):
    reference_text = words_to_attackedtext(words)
    for index, new_word in modification.items():
        transformed_text = reference_text.replace_word_at_index(index, new_word)
        # many textattack constraints only work for `WordSwap` transformation
        transformed_text.attack_attrs["last_transformation"] =  WordSwap()
    for C in constraints:
        if not C(transformed_text, reference_text):
            return False
    return True    

class GradientBasedGenerator:
    """ 
    Search text modifications in the vocabulary by a gradient x Embeddings way. 

    Args:

    model_bundle: `Model`, required
        used to fetch attributes, like embedding matrix, special ids.
    """

    def __init__(
        self, 
        model_bundle,
        max_swap: int=1,
        method: str = "linear_approx",
        constraints=[ WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech() ]
    ) -> None: 
        self.model_bundle = model_bundle
        self.max_swap = max_swap
        self.method = method
        self.constraints = constraints
        
    def generate_modifications(self, **kwargs):
        if self.method == "linear_approx":
            return self._generate_modifications_by_linear_approx(**kwargs)
        elif self.method == "grad_norm":
            return self._generate_positions_by_grad_norm(**kwargs)
        elif self.method == "random":
            return self._generate_positions_randomly(**kwargs)

    def _generate_modifications_by_linear_approx(self, words: List[str], grad, index_to_token, invalid_position=[], error_max=-1, max_swap=1, ): # (p, s) pairs
        """
        args:
        error_max: 1 for maximization, -1 for minimization
        """
        model_bundle = self.model_bundle
        token_start_idx = model_bundle.token_start_idx # TODO: ensure that special tokens are added during tokneization rather than indexing
        seq_len = len(words)
        token_end_idx = token_start_idx + (seq_len-1)
        num_vocab = len(index_to_token)
        
        # shape: seq_len * num_vocab
        _, indices = get_approximate_scores(
            grad[token_start_idx:token_end_idx, :], 
            model_bundle.embedding_matrix,
            all_special_ids=model_bundle.all_special_ids, 
            sign=error_max)
        # update modified postions and modifications
        modification = dict()
        idx_of_modify = 0  
        
        for idx_of_modify, index in enumerate(indices):
            if len(modification) >= max_swap:
                break
            position_to_flip, what_to_modify = int(index // num_vocab) + model_bundle.token_start_idx, int(indices[idx_of_modify] % num_vocab)
            idx_of_modify += 1
            if position_to_flip in modification.values(): # do not modify the same position twice in one iteration
                continue
            modify_token = index_to_token[int(what_to_modify)]
            if self.constraints:
                # validate modification by constraints
                if (not validate_word_swap(words, modification={position_to_flip: modify_token}, constraints = self.constraints)) or ( position_to_flip in invalid_position ):
                    continue
                
            modification[position_to_flip] = modify_token
        return modification

    def _generate_positions_by_grad_norm(self, words, grad, invalid_position=[], **kwargs):
        token_start_idx = self.model_bundle.token_start_idx # TODO: ensure that special tokens are added during tokneization rather than indexing
        seq_len = len(words)
        token_end_idx = token_start_idx + (seq_len-1)

        # np.sqrt(np.array([g.dot(g) for g in valid_grads]))
        scores_pos_to_modify = np.linalg.norm(grad[token_start_idx:token_end_idx], axis=1)
        positions_to_flip = np.argsort(scores_pos_to_modify)[::-1]
        for position_to_flip in positions_to_flip:
            if position_to_flip in invalid_position:
                continue
            return  {int(position_to_flip): None}

    def _generate_positions_randomly(self, words, invalid_position, **kwargs):
            len_tokens = len(words)
            
            index_order = np.arange(len_tokens)
            np.random.shuffle(index_order)
            for position_to_flip in index_order:
                if position_to_flip in invalid_position:
                    continue
                return  {int(position_to_flip): None}


"""
Modificatoin Applicator
============================================
"""           
class ModificationApplicator:
    def __init__(self, type, field_to_modify="tokens") -> None:
        if type != "squad":
            type="single_sentence_classification"
        self.type = type
        self.field_to_modify = field_to_modify

    def apply_modifications(self, instances, modifications):
        assert len(instances) == len(modifications)
        # create new batch so the original texts would not be modified
        batch_copy = deepcopy(instances)
        for instance, modification in zip(batch_copy, modifications):
            # change text field of each instance
            self.apply_modificatoin_on_single_instance( instance, modification)
        return batch_copy
    
    def apply_modificatoin_on_single_instance(self, instance, modification):
        for position_to_modify, substitution in modification.items(): 
            if self.type == "single_sentence_classification":
                instance.fields[self.field_to_modify].tokens[position_to_modify] = Token(substitution )
            elif self.type == "squad":
                passage_str = self._apply_modification_on_squad(instance, position_to_modify, substitution)
            else: 
                raise Exception


    def _apply_modification_on_squad(
            self,
            instance: Instance, \
            position_to_modify:int, \
            substitution: str,
            ):
            
        instance.fields["passage"].tokens[position_to_modify] = Token(substitution )

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
from allennlp_extra.data.dataset_readers import ClassificationFromJson

@DatasetReader.register("perturb_labeled_text")
class PerturbLabeledTextDatasetReader(ClassificationFromJson):
    def __init__(
            self, 
            mod_reader: DatasetReader,
            modification_path: str = None,
            fix_substitution: str = None,
            random_postion: bool = False,
            triggers: dict = None, # fix_insertion
            position: str = None,
            perturb_prob: float = 1.0,
            max_perturbed_instances: int = None,
            **kwargs,):
        super().__init__(**kwargs)
        # dataset_reader used to generate modifications
        self._mod_reader = mod_reader
        
        if os.path.exists(modification_path):
            with open(modification_path, 'rb') as file:
                self.modifications = pickle.load(file)
        else:
            self.modifications = None
        
        if triggers:
            assert isinstance(triggers, dict) # only for classification, not for rc
            for label, trigger_txt in triggers.items():
                triggers[label] = [Token(t) for t in trigger_txt]
        self.triggers = triggers
        self.perturb_prob = perturb_prob
        if self.perturb_prob != 1 and max_perturbed_instances is not None:
            raise Exception("`perturb_prob` and `max_perturbed_instances` are mutually exclusive.")
        self.max_perturbed_instances = max_perturbed_instances
        self.position = position
        self.fix_substitution = fix_substitution
        self.random_position = random_postion
                

    @overrides
    def _read(self, file_path):
        perturb_idx = 0

        for instance in self._mod_reader._read(file_path):
            
            if (self.max_perturbed_instances is None or \
                    perturb_idx < self.max_perturbed_instances) and \
                    random.uniform(0,1) <= self.perturb_prob : 
                logger.info(f'perurb {perturb_idx}')

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
        self._mod_reader.apply_token_indexers(instance)

    def perturb_labeled_single_sent(self, instance, perturb_idx):
        label = instance.fields['label'].label
        tokens = instance.fields['tokens'].tokens
        
        if self.modifications is not None:  
            # to correctly reconstruct the perturbed texts,
            # ensure that tokenizer is the same as the one generating modificaitons.
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
        mod_applicator = ModificationApplicator(type="squad", field_to_modify="passage")
        mod_applicator.apply_modificatoin_on_single_instance(instance, {position_to_modify: substitution})
       
        return instance

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
