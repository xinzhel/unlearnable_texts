import json
import logging
import os
from copy import Error, deepcopy
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from allennlp.common import Tqdm, cached_transformers
from allennlp.common import util as common_util
from allennlp.common.registrable import Registrable
from allennlp.data import Batch, DataLoader, Instance, Token
from allennlp.data.data_loaders import data_loader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import (ELMoTokenCharactersIndexer,
                                          SingleIdTokenIndexer,
                                          TokenCharactersIndexer)
from allennlp.models.model import Model
from allennlp.modules import token_embedders
from allennlp.modules.token_embedders import token_embedder
from allennlp.modules.token_embedders.pretrained_transformer_embedder import \
    PretrainedTransformerEmbedder
from allennlp.nn.util import (find_embedding_layer, find_text_field_embedder,
                              move_to_device)
from allennlp_models.rc import BidirectionalAttentionFlow
from torch import Tensor, backends
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.utils.hooks import RemovableHandle

from allennlp_extra.models.bart import Bart
from allennlp_extra.models.seq2seq import MySeq2Seq

from .constraints import (ExcludeAnswerSpan, PartOfSpeech, ProperNoun,
                          UniversalSentenceEncoder, WordEmbeddingDistance)

logger = logging.getLogger(__name__)

class TextModifier(Registrable):
    """ 
    Generate text modifications in a gradient-directed way. Specifically, this class aims to:
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
    default_implementation = "default"

    def __init__(
        self, 
        model: Model,
        data_loader: DataLoader,
        serialization_dir: str,
        perturb_bsz: int=32,
        constraints: List[str]=[],
        input_field_name: str = "tokens",
        max_swap: int=1,
        class_wise: bool =False,
        only_where_to_modify: bool = False,
        num_train_steps_per_perturbation: int = 10,
        error_max: int = -1, # 1 for maximization, -1 for minimization
        **kwargs, # it is important to add this to avoid errors with extra arguments from command lines for constructing other classes
    ) -> None:        
        self.perturb_bsz = perturb_bsz
        self.model = model
        self._vocab = self.model.vocab
        self.data_loader = data_loader
        self.batch_size = self.data_loader.batch_sampler.batch_size
        self._serialization_dir = serialization_dir
        self.indices_of_token_to_modify = only_where_to_modify
        self.num_train_steps_per_perturbation = num_train_steps_per_perturbation
        # just to ensure _instances is loaded
        self.instances = list(self.data_loader.iter_instances())
        self.num_examples = len(self.instances)
        
        self.error_max = error_max

        # constraints
        self.position_to_modify_constraints = []
        self.substitution_constraints = []
        
        for C in constraints:
            if C == "pos":
                self.substitution_constraints.append(PartOfSpeech())
            elif C == 'counter-fitting':
                self.substitution_constraints.append(WordEmbeddingDistance())
            elif C == 'sentence-encoder':
                self.substitution_constraints.append(UniversalSentenceEncoder())
            elif C=="PROPN":
                self.position_to_modify_constraints.append(ProperNoun())
            else:
                raise Error(f'The constraint {C} is not supported.')

        
        self.input_field_name = input_field_name

        if self.input_field_name == "passage":
            self.position_to_modify_constraints.append(ExcludeAnswerSpan())
        
        # special ids
        try:
            token_embedder = find_text_field_embedder(model).token_embedder_tokens
            if isinstance(token_embedder, PretrainedTransformerEmbedder):
                
                self.namespace = "tags"
                # get special ids
                model_name = token_embedder.transformer_model.config._name_or_path
                tokenizer = cached_transformers.get_tokenizer(model_name)
                self.all_special_ids = tokenizer.all_special_ids
            else:
                self.namespace = "tokens"
                # special ids
                self.all_special_ids = [
                    self._vocab._token_to_index[self.namespace][self._vocab._padding_token],
                    self._vocab._token_to_index[self.namespace][self._vocab._oov_token]
                ]
        except ValueError:
            if isinstance(model, Bart):
                self.namespace = "tokens"
                model_name = "facebook/" + model.bart.config._name_or_path
            else:
                raise Exception("Unsupported Model!")
            tokenizer = cached_transformers.get_tokenizer(model_name)
            self.all_special_ids = tokenizer.all_special_ids
        
        # find_valid_start_pos
        self.token_start_idx = 0
        if isinstance(model, Bart) or isinstance(model, MySeq2Seq) or isinstance(find_text_field_embedder(model).token_embedder_tokens, PretrainedTransformerEmbedder):
            # exclude the first special token [CLS] for BERT or <s> for BART
            self.token_start_idx = 1 


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

        # get embedding layer
        if isinstance(self.model, Bart):
            embedding_layer = self.model.bart.model.shared
        else:
            embedding_layer = find_embedding_layer(self.model)
        if isinstance(embedding_layer, 
            (token_embedders.embedding.Embedding, 
                torch.nn.Embedding, 
                torch.nn.modules.sparse.Embedding)):
            # If we're using something that already has an only embedding matrix, we can just use
            # that and bypass this method.
            self.embedding_matrix =  embedding_layer.weight
        else:
            all_tokens = list(self._vocab._token_to_index[self.namespace])
            inputs = self._make_embedder_input(all_tokens)

            # pass all tokens through the fake matrix (i.e., embedding layer) and create an embedding out of it.
            self.embedding_matrix = embedding_layer(inputs).squeeze()

        self.embedding_layer = embedding_layer

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
        
        grads, _ = get_grad(all_instances, self.model, self.embedding_layer, batch_size=self.batch_size) 
        num_vocab = self.embedding_matrix.size()[0]

        idx_of_instance = 0
        
        for idx_of_batch, instance in enumerate(all_instances):
            
            tokens = instance.fields[self.input_field_name].tokens
            seq_len = len(tokens)
            token_end_idx = self.token_start_idx + (seq_len-1)
            
            # shape: seq_len * num_vocab
            _, indices = \
                    get_approximate_scores(grads[idx_of_batch, self.token_start_idx:token_end_idx, :], self.embedding_matrix, \
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
                    modify_token = self._vocab._index_to_token[self.namespace][int(what_to_modify)]

                    # constraints
                    qualified = True
                    for C in self.position_to_modify_constraints:
                        qualified = C.check(position_to_flip, instance, self.input_field_name)
                    for C in self.substitution_constraints:
                        origin_token = tokens[position_to_flip].text
                        qualified = C.check(str(modify_token), str(origin_token))
                        if not qualified:
                            break
                    if not qualified:
                        continue
                    
                    modifications.append((position_to_flip, modify_token))
                    positions_flipped.append(position_to_flip)
            
            modification_dict = {}
            for position_to_flip, modify_token in modifications:
                modification_dict[position_to_flip] = modify_token
            self.modifications[idx_of_instance] = modification_dict
            idx_of_instance += 1


    def update_triggers(self, epoch, batch_idx):
        # always maintain a clean version of `self.instances`
        all_instances = deepcopy(self.instances)
        instances_dict = {key: [] for key in self._vocab._token_to_index['labels']}
        for instance in all_instances:
            instances_dict[instance['label'].label].append(instance)

        output = '\n Epoch: ' + str(epoch) + ' || Batch: '+str(batch_idx) + '\n'
        
        for label, instances in instances_dict.items():
            instances = prepend_batch(instances, self.triggers, self._vocab, self.input_field_name)
            
            lowest_loss = 9999
            num_no_improvement = 0
            patient = 10
            for batch_indices in self.data_loader.batch_sampler.get_batch_indices(instances):
                batch = []
                for i in batch_indices:
                    batch.append(instances[i])
                
                grads, loss = get_grad(batch, self.model, self.embedding_layer) 
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
        
        with open( os.path.join(self._serialization_dir, f"triggers.json"), 'a') as fp:
            fp.write(output)

    
    def update_positions_to_modify(self, wir_method='gradient'):
        """ Somewhat imitate the implementation from 
        `textattack.GreedyWordSwapWIR._get_index_order()`
        """
        # always maintain a clean version of `self.instances`
        all_instances = deepcopy(self.instances)
        assert len(self.modifications) == len(all_instances)
        if wir_method == "gradient":
            grads, _ = get_grad(all_instances, self.model, self.embedding_layer)
            for idx_of_instance, instance in enumerate(all_instances):
                
                tokens = instance.fields[self.input_field_name].tokens
                scores_pos_to_modify = [g.dot(g) for g in grads[idx_of_instance][:len(tokens),:]]
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

        # TODO: correctly saving modifications into file when max_swap > 1
        assert self.max_swap == 1

        # construct file name
        file_name = ''
        if self.indices_of_token_to_modify:
            file_name += 'pos-'
        if self.error_max == 1:
            file_name += 'error_max'
        if epoch is None:
            file_name += "init_modification"
        else:
            assert batch_idx is not None
            file_name += f"modification_epoch{epoch}_batch{batch_idx}"

        
        # TODO: the following process should be applied for all other datasets
        if save_text and self.input_field_name == "passage":
            final_output = {}
            # add modified texts into modifications
            all_instances = deepcopy(self.instances)
            for instance, modification in zip(all_instances, self.modifications):
                output_dict = {}
                # origin info: passage, question, answer
                output_dict['orig_passage'] = instance.fields['metadata'].metadata["original_passage"]
                output_dict['question'] = " ".join(instance.fields['metadata'].metadata["question_tokens"])
                ## answer
                answer_start = instance.fields['span_start'].sequence_index
                answer_end = instance.fields['span_end'].sequence_index
                output_dict['answer_text'] = instance.fields['metadata'].metadata["answer_texts"]
                # answer_tokens =  instance.fields['passage'].tokens[answer_start:answer_end+1]
                # output_dict['answer_text'] = " ".join([str(token) for token in answer_tokens])
                output_dict['answer_start_position'] = answer_start
                output_dict['answer_end_position'] = answer_end   

                # info after modification
                position_to_modify, substitution = list(modification.items())[0]
                output_dict['modified_position'] = int(position_to_modify)
                output_dict['modified_word'] = str(instance.fields['passage'].tokens[int(position_to_modify)])
                output_dict['substitution_word']= substitution
                self.modify_one_example(instance, int(position_to_modify), substitution, self.input_field_name)
                output_dict['modified_passage'] = instance.fields['metadata'].metadata["original_passage"]
                output_dict['distance_to_answer'] = min(abs(answer_start-int(position_to_modify)),abs(answer_end -int(position_to_modify)))
            
                id = instance.fields['metadata'].metadata["id"]
                final_output[id] = output_dict
        
            # save modifications
            with open( os.path.join(self._serialization_dir, file_name+'.json'), 'w') as fp:
                json.dump(final_output, fp)
        elif save_text  and self.input_field_name == "source_tokens":
            final_output = {}
            # add modified texts into modifications
            all_instances = deepcopy(self.instances)
            for instance, modification in zip(all_instances, self.modifications):
                output_dict = {}
                
                # info after modification
                position_to_modify, substitution = list(modification.items())[0]
                output_dict['modified_position'] = int(position_to_modify)
                output_dict['modified_word'] = str(instance.fields['source_tokens'].tokens[int(position_to_modify)])
                output_dict['substitution_word']= substitution
                modified_article = self.modify_one_example(instance, int(position_to_modify), substitution, self.input_field_name)
                output_dict['modified_article'] = modified_article

                id = instance.fields['metadata'].metadata["id"]
                final_output[id] = output_dict
            # save modifications
            with open( os.path.join(self._serialization_dir, file_name+'.json'), 'w') as fp:
                json.dump(final_output, fp)
            
        else:
            with open( os.path.join(self._serialization_dir, file_name+'.json'), 'w') as fp:
                json.dump(self.modifications, fp)


    
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

    def _make_embedder_input(self, all_tokens: List[str]) -> Dict[str, torch.Tensor]:
        """ copy from 
        https://github.com/allenai/allennlp/blob/main/allennlp/interpret/attackers/hotflip.py
        """
        inputs = {}
        
        # A bit of a hack; this will only work with some dataset readers, but it'll do for now.
        indexers = self.predictor._dataset_reader._token_indexers  # type: ignore
        for indexer_name, token_indexer in indexers.items():
            if isinstance(token_indexer, SingleIdTokenIndexer):
                all_indices = [
                    self.vocab._token_to_index[self.namespace][token] for token in all_tokens
                ]
                inputs[indexer_name] = {"tokens": torch.LongTensor(all_indices).unsqueeze(0)}
            elif isinstance(token_indexer, TokenCharactersIndexer):
                tokens = [Token(x) for x in all_tokens]
                max_token_length = max(len(x) for x in all_tokens)
                # sometime max_token_length is too short for cnn encoder
                max_token_length = max(max_token_length, token_indexer._min_padding_length)
                indexed_tokens = token_indexer.tokens_to_indices(tokens, self.vocab)
                padding_lengths = token_indexer.get_padding_lengths(indexed_tokens)
                padded_tokens = token_indexer.as_padded_tensor_dict(indexed_tokens, padding_lengths)
                inputs[indexer_name] = {
                    "token_characters": torch.LongTensor(
                        padded_tokens["token_characters"]
                    ).unsqueeze(0)
                }
            elif isinstance(token_indexer, ELMoTokenCharactersIndexer):
                elmo_tokens = []
                for token in all_tokens:
                    elmo_indexed_token = token_indexer.tokens_to_indices(
                        [Token(text=token)], self.vocab
                    )["elmo_tokens"]
                    elmo_tokens.append(elmo_indexed_token[0])
                inputs[indexer_name] = {"elmo_tokens": torch.LongTensor(elmo_tokens).unsqueeze(0)}
            else:
                raise RuntimeError("Unsupported token indexer:", token_indexer)

        return inputs


    @classmethod
    def generate_squad_analyzable_result(cls, instances, modification_file_path, print_num=None):
        """
        The example of the modification format: [{"80": "the"}, {"76": "the"}, {"22": "the"}]

        # Parameters

        instances: `Iterator[Instance]`, required
        
        modification_file_path: `str`, required

        # return

        """
        with open( modification_file_path, 'r') as fp:
            modifications: List[dict] = json.load(fp)

        print_results = []
        for idx_of_dataset, modification_dict in enumerate(modifications):
            instance = instances[int(idx_of_dataset)]
            fields = instance.fields
            print_result = {'original': " ".join(str(token) for token in fields["passage"].tokens)}

            for position_to_modify, substitution in modification_dict.items():
                
                print_result['position_to_modify'] = int(position_to_modify)
                print_result['modified_word'] = str(fields['passage'].tokens[int(position_to_modify)])
                print_result['substitution_word'] = substitution
                print_result['question'] = " ".join([str(token) for token in fields['question'].tokens])

                answer_start = fields['span_start'].sequence_index
                answer_end = fields['span_end'].sequence_index
                print_result['answer_text'] = fields['metadata'].metadata["answer_texts"]
                # answer_tokens =  fields['passage'].tokens[answer_start:answer_end+1]
                # print_result['answer_text'] = " ".join([str(token) for token in answer_tokens])
                print_result['position_answer_start'] = answer_start
                print_result['position_answer_end'] = answer_end
                print_result['distance_to_answer'] = min(abs(answer_start-int(position_to_modify)),abs(answer_end -int(position_to_modify)))
                
                
                
                break # only modify one position
            print_results.append(print_result)
            if print_num is not None and idx_of_dataset >=print_num:
                break
        return print_results

# we inspect from_param would use default __init__ constructor
TextModifier.register("default", constructor=None)(TextModifier)

def prepend_batch(instances, trigger_tokens, vocab, input_field_name="tokens"):
    """
    trigger_tokens List[str] ：
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

def get_approximate_scores(grad, embedding_matrix, all_special_ids: List[int] =[], sign: int = -1):
    """ The objective is to minimize first-order approximate of L(replace_text):
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

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_grad(
    instances: Iterable[Instance], 
    model: torch.nn.Module, 
    layer: torch.nn.Module, 
    loss_fct: _Loss = None,
    batch_size: int = 16):
    """ 
    # Parameters

    batch: A batch of instance
    model: (1) the subclass of the `PreTrainedModel`  or 
           (2) Pytorch model with a method "get_input_embeddings" which return `nn.Embeddings`
    layer: the layer of `model` to get gradients, e.g., a embedding layer
    batch_size: avoid the case that `instances` may be too overloaded to perform forward/backward pass

    # Return

    return_grad: shape (batch size, sequence_length, embedding_size): gradients for all tokenized elements
        , including the special prefix/suffix and <SEP>.
    """
    
    cuda_device = next(model.parameters()).device

    # register hook
    gradients: List[Tensor] = []
    hooks: List[RemovableHandle] = _register_gradient_hooks(gradients, layer)

    # require grads for all model params 
    original_param_name_to_requires_grad_dict = {}
    for param_name, param in model.named_parameters():
        original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
        param.requires_grad = True

    # calculate grad for inference network
    orig_mode = model.training
    model.train(mode=True)
 
    # To bypass "RuntimeError: cudnn RNN backward can only be called in training mode"
    with backends.cudnn.flags(enabled=False):
        
        all_batches = Batch(instances)
        all_batches.index_instances(model.vocab)
        dataset_tensor_dict = all_batches.as_tensor_dict()
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
