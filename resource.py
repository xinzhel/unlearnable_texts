# coding=utf-8
# Copyright 2022 Xinzhe Li All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# from functools import cache
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass
from pydantic import NoneStr
from torch.nn import Module, CosineSimilarity
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import Union, List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
import nltk
from nltk.corpus import opinion_lexicon
import random
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.common.util import lazy_groups_of
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
from allennlp.data import Vocabulary
from allennlp_models.rc import BidirectionalAttentionFlow
from allennlp_models.rc.dataset_readers import TransformerSquadReader


"""
IO utils
============================================
"""
def find_load_folder(cur_dir):
    """ find a folder called data from the project repository to parent(or parents') repositories
    """
    # successfully exit
    if os.path.exists(os.path.join(cur_dir, 'data')):
        return os.path.join(cur_dir, 'data')
    # fail exit
    par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
    if par_dir == cur_dir: # root dir
        return None
    
    # recursive call
    return find_load_folder(par_dir)

"""
Resource Abstract Class
============================================
"""
class LoadedResource(ABC):
    """ An abstract class for loading resource (e.g., datasets, Pytorch models)
    """
    def __init__(self):
        self.loaded_resource = dict()
        self.load_folder = find_load_folder(os.getcwd())

    def __getitem__(self, key, reload=False):
        if key not in self.loaded_resource.keys() or reload:
            self.loaded_resource[key] = self._load(key)

        return self.loaded_resource[key]

    def __setitem__(self, key, value):
        self.loaded_resource[key] = value

    @abstractmethod
    def _load(self, name):
        """ Returns loaded resource
        """
        raise NotImplementedError

"""
LoadedDatasets Class
============================================
"""
class LoadedDatasets(LoadedResource):
    shuffle: bool = False
    def _load(self, dataset_name):
        if dataset_name == "dbpedia14":
            assert self.load_folder is not None
            dataset = load_dataset("csv", column_names=["label", "title", "sentence"],
                                    data_files={"train": os.path.join(self.load_folder, "dbpedia_csv/train.csv"),
                                                "validation": os.path.join(self.load_folder, "dbpedia_csv/test/test.csv")})
            dataset = dataset.map(self.target_offset, batched=True)
            num_labels = 14
        elif dataset_name == "ag_news":
            dataset = load_dataset("ag_news")
            num_labels = 4
        #################### binary sentiment classification #################### 
        elif dataset_name == "imdb":
            dataset = load_dataset("imdb", ignore_verifications=True)
            num_labels = 2
        elif dataset_name == "yelp_long":
            dataset = load_dataset("yelp_polarity")
            num_labels = 2
        elif dataset_name == "yelp":
            dataset_dict = dict()
            for split in ['train', 'val', 'test']:
                
                with open(os.path.join(self.load_folder, f"yelp/{split}/data.txt")) as file:
                    texts = [line.rstrip() for line in file]
                with open(os.path.join(self.load_folder, f"yelp/{split}/labels.txt")) as file:
                    labels = [int(line.rstrip()) for line in file]
                dataset_dict[split] = Dataset.from_dict({'text': texts, 'label':labels})
            dataset = DatasetDict(dataset_dict)
            num_labels = 2
        ############################### GLUE ###############################
        elif dataset_name == "sst2":
            assert self.load_folder is not None
            dataset = load_dataset(
                "csv", 
                column_names=["text", "label"],
                data_files={"train": os.path.join(self.load_folder, "sst/train.csv"),
                    "dev": os.path.join(self.load_folder , "sst/dev.csv"),
                    "test": os.path.join(self.load_folder, "sst/test.csv"),}
                )
            num_labels = 2
        elif dataset_name == "mnli":
            dataset = load_dataset("glue", "mnli")
            num_labels = 3
        elif dataset_name == "mrpc":
            dataset = load_dataset('glue', 'mrpc')
            num_labels = 2
        ############################### QA ###############################
        # dowload the QA data from https://github.com/michiyasunaga/LinkBERT
        elif dataset_name == "squad":
            pass
        elif dataset_name == "hotpot_qa":
            pass
        elif dataset_name == "trivia_qa":
            pass
        elif dataset_name == "natural_questions":
            pass
        elif dataset_name == "news_qa":
            pass
        elif dataset_name == "search_qa":
            pass
        else:
            raise Exception("Cannot find the dataset.")

        if self.shuffle:
            dataset = dataset.shuffle(seed=0)
        
        return dataset, num_labels

    # offset target by 1 if labels start from 1
    @staticmethod
    def target_offset(examples):
        examples["label"] = list(map(lambda x: x - 1, examples["label"]))
        return examples

datasets = LoadedDatasets()

"""
Available models in huggingface hub
============================================
"""
# transformers
# TODO: for consistency, use the name composed of `{PLM_name}-{dataset_name}`
hf_lm_names = {
    'bert-base-uncased': 'bert-base-uncased',
    'roberta-base': 'roberta-base',
    'albert-base-v2': 'albert-base-v2',

    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2'
}
hf_model_names =  {
    # SQUAD
    'bert-base-uncased-squad2': 'deepset/bert-base-uncased-squad2',
    'roberta-base-squad2': 'deepset/roberta-base-squad2',
    'albert-base-v2-squad2': 'twmkn9/albert-base-v2-squad2',

    # for SST-2
    'bert-base-uncased-SST-2': 'textattack/bert-base-uncased-SST-2',
    'roberta-base-SST-2': 'textattack/roberta-base-SST-2',
    'albert-base-v2-SST-2':'textattack/albert-base-v2-SST-2',
    # consistent with dataset_name
    'bert-base-uncased-sst2': 'textattack/bert-base-uncased-SST-2',
    'bert-sst2': 'textattack/bert-base-uncased-SST-2',
    'roberta-base-sst2': 'textattack/roberta-base-SST-2',
    'roberta-sst2': 'textattack/roberta-base-SST-2',
    'albert-base-v2-sst2':'textattack/albert-base-v2-SST-2',
    'albert-sst2':'textattack/albert-base-v2-SST-2',
    # 'textattack/distilbert-base-uncased-SST-2',
    # 'textattack/distilbert-base-cased-SST-2',
    # 'textattack/xlnet-base-cased-SST-2',
    # 'textattack/xlnet-large-cased-SST-2',
    # 'textattack/facebook-bart-large-SST-2',

    # for yelp
    'bert-base-uncased-yelp': 'textattack/bert-base-uncased-yelp-polarity',
    'roberta-base-yelp': 'VictorSanh/roberta-base-finetuned-yelp-polarity',
    'albert-base-v2-yelp':'textattack/albert-base-v2-yelp-polarity',
    'bert-yelp': 'textattack/bert-base-uncased-yelp-polarity',
    'roberta-yelp': 'VictorSanh/roberta-base-finetuned-yelp-polarity',
    'albert-yelp':'textattack/albert-base-v2-yelp-polarity',

    # for ag-news
    'bert-base-uncased-ag-news': 'textattack/bert-base-uncased-ag-news',
    'roberta-base-ag-news': 'textattack/roberta-base-ag-news', 
    'albert-base-v2-ag-news': 'textattack/albert-base-v2-ag-news',
    # consistent with dataset_name
    'bert-base-uncased-ag_news': 'textattack/bert-base-uncased-ag-news',
    'roberta-base-ag_news': 'textattack/roberta-base-ag-news', 
    'albert-ag_news': 'textattack/albert-base-v2-ag-news',
    'bert-ag_news': 'textattack/bert-base-uncased-ag-news',
    'roberta-ag_news': 'textattack/roberta-base-ag-news', 
    'albert-ag_news': 'textattack/albert-base-v2-ag-news',
    # 'textattack/distilbert-base-uncased-ag-news',

    # for MRPC
    'bert-base-uncased-MRPC': 'textattack/bert-base-uncased-MRPC',
    'roberta-base-MRPC': 'textattack/roberta-base-MRPC',
    'albert-base-v2-MRPC': 'textattack/albert-base-v2-MRPC',
    # consistent with dataset_name
    'bert-base-uncased-mrpc': 'textattack/bert-base-uncased-MRPC',
    'roberta-base-mrpc': 'textattack/roberta-base-MRPC',
    'albert-base-v2-mrpc': 'textattack/albert-base-v2-MRPC',

    # for QQP
    'bert-base-uncased-QQP': 'textattack/bert-base-uncased-QQP',
    # 'textattack/distilbert-base-uncased-QQP',
    # 'textattack/distilbert-base-cased-QQP',
    'albert-base-v2-QQP': 'textattack/albert-base-v2-QQP',
    'roberta-base-QQP': 'howey/roberta-large-qqp',
    # 'textattack/xlnet-large-cased-QQP',
    # 'textattack/xlnet-base-cased-QQP',

    # for snil
    # 'textattack/bert-base-uncased-snli',
    # 'textattack/distilbert-base-cased-snli',
    # 'textattack/albert-base-v2-snli',

    # for WNLI
    # 'textattack/bert-base-uncased-WNLI',
    # 'textattack/roberta-base-WNLI',
    # 'textattack/albert-base-v2-WNLI',

    # for MNLI
    # 'textattack/bert-base-uncased-MNLI',
    # 'textattack/distilbert-base-uncased-MNLI',
    # 'textattack/roberta-base-MNLI',
    # 'textattack/xlnet-base-cased-MNLI',
    # 'textattack/facebook-bart-large-MNLI',
    # 'facebook/bart-large-mnli',
}
    

"""
LoadedHfModels Class
============================================
"""
class LoadedHfModels(LoadedResource):

    def _load(self, name): 
        if name in hf_model_names.keys():
            model = AutoModelForSequenceClassification.from_pretrained(hf_model_names[name])
        if name in hf_lm_names.keys():
            model = AutoModelForMaskedLM.from_pretrained(hf_lm_names[name])
        return model

hf_models = LoadedHfModels()

"""
LoadedHfTokenizers Class
============================================
"""
class LoadedHfTokenizers(LoadedResource):

    def _load(self, name):
        all_names = {**hf_model_names, **hf_lm_names}
        print('Valid name:', name)
        assert name in all_names
        valid_name = all_names[name]
            
        return AutoTokenizer.from_pretrained(valid_name)
       

hf_tokenizers = LoadedHfTokenizers()

"""
Model-Tokenizer Bundle Class
============================================
"""
@dataclass
class HfModelBundle:
    model_name: str
    model: Module
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    @property
    def embedding_layer(self):
        return self.model.get_input_embeddings()
    
    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def full_model_name(self):
        all_names = {**hf_lm_names, **hf_model_names}
        if self.model_name in all_names:
            return all_names[self.model_name]
        return None

    @property
    def all_special_ids(self):
        return self.tokenizer.all_special_ids

    def tokenize_from_words(self, words1: List[str], words2: List[str]=None):
        if getattr(self, "allennlp_tokenizer", None) is None:
            self.allennlp_tokenizer = PretrainedTransformerTokenizer(self.full_model_name)
        if words2 is None:
            wordpieces, offsets = self.allennlp_tokenizer.intra_word_tokenize(words1)
        else:
            wordpieces, offsets, offsets2 = self.allennlp_tokenizer.intra_word_tokenize_sentence_pair(words1, words2)
            offsets.extend(offsets2)
        wordpieces = {
            'token_str': [t.text for t in wordpieces],
            "input_ids": [t.text_id for t in wordpieces],
            "token_type_ids": [t.type_id for t in wordpieces],
            "attention_mask": [1] * len(wordpieces), 
        }
        
        return wordpieces, offsets
    
    def get_logit(self, words):
        model_output = self.get_model_output(words, y=None)
        if isinstance(model_output, SequenceClassifierOutput):
            return model_output.logits[0]
        else: 
            return None
            
    def get_model_output(self, words, y=None):
        wordpieces = self.tokenize_from_words(words)[0]
        model_input = {
            "input_ids": torch.LongTensor(wordpieces['input_ids']).unsqueeze(0),
            "token_type_ids": torch.LongTensor(wordpieces['token_type_ids']).unsqueeze(0),
            "attention_mask": torch.LongTensor(wordpieces['attention_mask']).unsqueeze(0), 
            
        }
        if y is not None:
            model_input["labels"] = torch.LongTensor([y]).unsqueeze(0)
        model_output = self.forward(model_input)
        return model_output    

    def forward(self, model_input, return_last_hidden_states=False):
        # to correct device
        device = self.device
        model_input = {k: v.to(device) for k, v in model_input.items()}
        self.model.to(device)

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**model_input, output_hidden_states=True) 

        if return_last_hidden_states:
            # TODO: include other types of tasks
            assert isinstance(model_output, (SequenceClassifierOutput, MaskedLMOutput))

            # hidden_state[-1] taking the output of the last encoding layer from 
            # tuple for all layers' the hidden state
            last_hidden_state = model_output.hidden_states[-1]

            return last_hidden_state
            
        else:
            return model_output

    def get_sentence_embedding(self, examples: List[str], use_cls=False, normalize=False):
        embeddings = []
        for batch in lazy_groups_of(examples, 2): # assume the device is capable to deal with bsz 2
            model_input = self.tokenizer.batch_encode_plus(
                batch, 
                max_length=256, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
                )
            
            last_hidden_states = self.forward(model_input, return_last_hidden_states=True)
            if use_cls:  
                # [:, 0]  taking the hidden state corresponding
                # to the first token, i.e., [CLS].           
                embeddings.append(last_hidden_states[:, 0])
            else:
                all_tokens_embeddings = last_hidden_states[:, 1:]
                embeddings.append(all_tokens_embeddings.mean(dim=1))
        result = torch.cat(embeddings, dim=0) # dim: [num_sample, hidden_size]
        if normalize:
            return (result - result.mean(dim=1))/result.std(dim=1)
        else:
            return result
    
    def get_sentence_embedding_from_words(self, words, use_cls=False, normalize=False):
        wordpieces = self.tokenize_from_words(words)[0]
        model_input = {
            "input_ids": torch.LongTensor(wordpieces['input_ids']).unsqueeze(0),
            "token_type_ids": torch.LongTensor(wordpieces['token_type_ids']).unsqueeze(0),
            "attention_mask": torch.LongTensor(wordpieces['attention_mask']).unsqueeze(0), 
        }
        last_hidden_states = self.forward(model_input, return_last_hidden_states=True)
        if use_cls:
            result = last_hidden_states[:, 0] # dim: [num_sample=1, hidden_size]
        else:
            all_tokens_embeddings = last_hidden_states[:, 1:] # dim: [num_sample=1, seq_len, hidden_size]
            result =  all_tokens_embeddings.mean(dim=1) # dim: [num_sample=1, hidden_size]
        if normalize:
            return (result - result.mean(dim=1))/result.std(dim=1)
        else:
            return result

    def get_cos_sim(self, words1, words2):
        
        cos_sim = CosineSimilarity(dim=1)
        sent_emb1 = self.get_sentence_embedding_from_words(words1)
        sent_emb2 = self.get_sentence_embedding_from_words(words2)
        return cos_sim(sent_emb1, sent_emb2)

class LoadedHfModelBundle(LoadedResource):

    def _load(self, name):
        tokenizer = hf_tokenizers[name]
        model = hf_models[name]
        return HfModelBundle(name, model, tokenizer)

hf_model_bundles = LoadedHfModelBundle()

from functools import lru_cache

@dataclass
class AllenNLPModelBundle:
    model: Module
    vocab: Vocabulary

    @property
    @lru_cache()
    def all_special_ids(self):
        token_embedder = find_text_field_embedder(self.model).token_embedder_tokens
        if isinstance(token_embedder, PretrainedTransformerEmbedder):
            self.namespace = "tags"
            # get special ids
            model_name = token_embedder.transformer_model.config._name_or_path
            tokenizer = cached_transformers.get_tokenizer(model_name)
            return tokenizer.all_special_ids
        # elif isinstance(self.model, allennlp_extra.models.bart.Bart):
        #     self.namespace = "tokens"
        #     model_name = "facebook/" + self.model.bart.config._name_or_path
        #     tokenizer = cached_transformers.get_tokenizer(model_name)
        #     self.all_special_ids = tokenizer.all_special_ids
        else:
            self.namespace = "tokens"
            return [
                self.vocab._token_to_index[self.namespace][self.vocab._padding_token],
                self.vocab._token_to_index[self.namespace][self.vocab._oov_token]
            ]
 
    @property
    def namespace(self):
        token_embedder = find_text_field_embedder(self.model).token_embedder_tokens
        if isinstance(token_embedder, PretrainedTransformerEmbedder):
            return "tags"
        else:
            return "tokens"

    @property
    def token_start_idx(self):
        token_embedder = find_text_field_embedder(self.model).token_embedder_tokens
        if  isinstance(token_embedder, PretrainedTransformerEmbedder):
            # or isinstance(model, Bart) or isinstance(model, MySeq2Seq)
            return 1  # exclude the first special token [CLS] for BERT or <s> for BART
        else:
            return 0
    
    @property
    def embedding_layer(self):
        # if isinstance(self.model, Bart):
        #     return self.model.bart.model.shared
        return find_embedding_layer(self.model)

    @property
    def embedding_matrix(self):
        if isinstance(self.embedding_layer, 
            (token_embedders.embedding.Embedding, 
                torch.nn.Embedding, 
                torch.nn.modules.sparse.Embedding)):
            # If we're using something that already has an only embedding matrix, we can just use
            # that and bypass this method.
            return self.embedding_layer.weight
        else: # other types in `allennlp.module.token_enbedders`
            all_tokens = list(self.vocab._token_to_index[self.namespace])
            inputs = self._make_embedder_input(all_tokens)

            # pass all tokens through the fake matrix (i.e., embedding layer) and create an embedding out of it.
            return self.embedding_layer(inputs).squeeze()

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

"""
Sentiment Lexicon
============================================
"""
def get_sentiment_lexicon(n=50, simple_word=False):
    tagged_neg = nltk.pos_tag(list(opinion_lexicon.negative()))
    tagged_pos = nltk.pos_tag(list(opinion_lexicon.positive()))
    pos_lexicon = [] 
    neg_lexicon = []
    for word, tag in tagged_pos:
        if tag == 'JJ': # adjective
            pos_lexicon.append(word)

    for word, tag in tagged_neg:
        if tag == 'JJ':
            neg_lexicon.append(word)
    def is_simple_word(word):
        # hf_tokenizers['bert-base-uncased'].tokenize(word)
        return True

    if simple_word:
        pos_lexicon = [w for w in pos_lexicon if is_simple_word(w) ]
        neg_lexicon = [w for w in neg_lexicon if is_simple_word(w) ]

    random.seed(20)
    pos_lexicon50 = random.sample(pos_lexicon, n)
    random.seed(20)
    neg_lexicon50 = random.sample(neg_lexicon, n)
    return pos_lexicon50, neg_lexicon50
