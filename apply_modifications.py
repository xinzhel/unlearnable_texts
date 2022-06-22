import logging
import pickle
import timeit

from allennlp.common.util import import_module_and_submodules
from allennlp.common import Params
from allennlp.common import util as common_util
from typing import Any, Dict
from allennlp_models.classification import *
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from allennlp.common.util import int_to_device
from allennlp.training import util as training_util
from allennlp.nn.util import move_to_device
from allennlp.training.optimizers import AdamOptimizer
import utils
from utils import PerturbLabeledTextDatasetReader, PerturbedTransformerSquadReader
import resource
import pandas as pd
from textattack.constraints.grammaticality.part_of_speech import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance

task="ag_news"
model_name="lstm"
serialization_dir = f'outputs/{task}/{model_name}'
mod_file = "modifications_30"
param_path = f'config/{task}/{model_name}.jsonnet'
include_package = ['allennlp_extra']

if __name__=="__main__":

    # import my allennlp plugin for unlearnable
    for package_name in include_package:
        import_module_and_submodules(package_name)

    # construct model and dataloader
    params = Params.from_file(param_path) # parse JSON file
    common_util.prepare_environment(params)  # set random seed
   
    required_params = {k: params.params[k] for k in ['dataset_reader', 'train_data_path', 'data_loader']}
    object_constructor = utils.GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=serialization_dir)
    data_loader = object_constructor.data_loader
    with open(f'{serialization_dir}/{mod_file}.pickle', 'rb') as file:
        modifications = pickle.load(file)
    mod_applicator = utils.ModificationApplicator(type=task)

    train_instances = list(data_loader.iter_instances())
    modified_instances = mod_applicator.apply_modifications(train_instances, modifications)

    instances_dict = []
    for inst in  modified_instances:
        instances_dict.append({"label": int(inst.fields['label'].label), "text": ' '.join(inst.fields['tokens'].human_readable_repr())})
    df = pd.DataFrame(instances_dict)
    df.to_json(f"{serialization_dir}/train_{mod_file}.json", orient = "records", lines=True)