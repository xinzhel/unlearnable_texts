import argparse
import os
import logging
import resource
import pickle
from allennlp.models import load_archive
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.common.util import import_module_and_submodules
from allennlp.common import Params, Registrable, Lazy
from allennlp.common import util as common_util
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer
from typing import Any
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from allennlp.predictors import Predictor
import utils
from textattack.constraints.grammaticality.part_of_speech import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance

task="sst2"
model_name="lstm"
serialization_dir = f'outputs/{task}/{model_name}'
model_path = f'models/{task}/{model_name}' 
# squad bidaf: '../models/squad/bidaf_glove_well_trained/model.tar.gz'
cuda_device = 0

def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--param_path", 
        type=str,
        default=f'config/{task}/generate_unlearnable/{model_name}/{model_name}.jsonnet',
        help="path to parameter file describing the model to be trained"
    )

    parser.add_argument(
                    "--include-package",
                    type=str,
                    action="append",
                    default=['allennlp_extra'],
                    help="additional packages to include",
                )

    parser.add_argument(
        "-s",
        "--serialization-dir",
        default=serialization_dir,
        type=str,
        help="directory in which to save the model and its logs",
    )

    # Now we can parse the arguments.
    args = parser.parse_args()

    return parser, args

if __name__=="__main__":
    # parse command line arguments
    parser, args = parse_cmd_args()

    # import my allennlp plugin
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)

    # data_loader for training data
    parameter_filename=args.param_path
    params = Params.from_file(parameter_filename)
    required_params = {k: params.params[k] for k in ['dataset_reader', 'train_data_path', 'data_loader']}
    object_constructor = utils.GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=serialization_dir)
    data_loader = object_constructor.data_loader
    train_instances = list(data_loader.iter_instances())

    # model
    # predictor = Predictor.from_path(model_path)
    # model = predictor._model
    model = load_archive(model_path).model 
    model_bundle = resource.AllenNLPModelBundle(model, model.vocab)

    # generate error-max modification
    if task=="sst2":
        constraints = [WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech()]
    else:
        constraints = []
    mod_generator = utils.GradientBasedGenerator(model_bundle, constraints=constraints), 
    mod_applicator = utils.ModificationApplicator(type=task)

    all_modifications = utils.generate_modifications(
            train_instances, 
            model_bundle, 
            mod_generator, 
            mod_applicator,
            current_modifications=None, 
            invalid_positions=[], 
            field_to_modify="tokens", 
            vocab_namespace="tokens",
            error_max=1
    )

    with open(f'{serialization_dir}/modifications_error_max.pickle', 'wb') as file:
        pickle.dump( all_modifications, file)


   