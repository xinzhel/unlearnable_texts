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

task="squad"
model_name="bidaf_glove"
serialization_dir = f'outputs/{task}/{model_name}'
cuda_device = 0
num_epochs=1
mode = "min_min"
num_train_steps_per_perturbation=30 # set infinite numbers to only train the model
param_path = f'config/{task}/{model_name}.jsonnet'
include_package = ['allennlp_extra']


"""
generate error min modifications 
============================================
"""
def generate_mod(model_bundle, data_loader, test_data_loader, optimizer, mod_generator, mod_applicator,  num_train_steps_per_perturbation=30, only_batches_modifications=False, task=None) -> Dict[str, float]:
    """
    Trains one epoch
    """
    field_to_modify = "tokens"
    metric = "accuracy"
    if task == "squad":
        field_to_modify = "passage"
        metric = 'f1' #['start_acc', 'end_acc', 'span_acc', 'em', 'f1']

    train_loss = 0.0
    model.train()
    train_instances = list(data_loader.iter_instances())
    assert len(train_instances) <10000

    num_training_batches = len(data_loader)
    print(f'# of training batches : {str(num_training_batches)} ')
    batches_in_epoch_completed = 0
    batches_modifications = {}
    modification_batch_idx = 0
    list_of_batch_indices = list(data_loader.batch_sampler.get_batch_indices(train_instances))
    current_test_accuracy = None
    modifications_history = dict()
    for i, dataset_indices in enumerate(list_of_batch_indices):

        # generate unleanrable modifications
        if batches_in_epoch_completed % num_train_steps_per_perturbation == 0:
            # get current test accuracy
            test_metric = model_bundle.evaluate_from_dataloader(test_data_loader)[metric]
            if current_test_accuracy is None or test_metric > current_test_accuracy:
                current_test_accuracy = test_metric
            else:
                return all_modifications, modifications_history
               
            if not only_batches_modifications:
                all_modifications = utils.generate_modifications(
                        train_instances, 
                        model_bundle, 
                        mod_generator,
                        mod_applicator,
                        current_modifications=None, 
                        invalid_positions=[], 
                        field_to_modify=field_to_modify, 
                        vocab_namespace="tokens"
                    )
                modifications_history[batches_in_epoch_completed] = all_modifications
                with open(f'{serialization_dir}/modifications_{batches_in_epoch_completed}.pickle', 'wb') as file:
                    pickle.dump( all_modifications, file)
            else:
                batches_modifications = list()
                for j in range(num_train_steps_per_perturbation):
                    if i+1+j >= len(list_of_batch_indices):
                        all_modifications = utils.generate_modifications(
                            train_instances, 
                            model_bundle, 
                            mod_generator,
                            mod_applicator,
                            current_modifications=None, 
                            invalid_positions=[], 
                            field_to_modify=field_to_modify, 
                            vocab_namespace="tokens"
                        )
                        return all_modifications, modifications_history
                    next_batch = [train_instances[i] for i in list_of_batch_indices[i+1+j]]
                    # generate modifications for all the instances
                    batch_modifications = utils.generate_modifications(
                        next_batch, 
                        model_bundle, 
                        mod_generator,
                        mod_applicator,
                        current_modifications=None, 
                        invalid_positions=[], 
                        field_to_modify=field_to_modify, 
                        vocab_namespace="tokens"
                    )
                    batches_modifications.append(batch_modifications)
                modification_batch_idx = 0
                modifications_history[batches_in_epoch_completed] = batches_modifications
        
 
        batch = [train_instances[i] for i in dataset_indices]
        ##### apply unleanrable modifications
        if not only_batches_modifications: 
            batch_modifications = [all_modifications[i] for i in dataset_indices]
        else:
            assert len(batches_modifications) == num_train_steps_per_perturbation
            assert modification_batch_idx < num_train_steps_per_perturbation
            batch_modifications = batches_modifications[modification_batch_idx]
            modification_batch_idx += 1    
        
        batch = mod_applicator.apply_modifications(batch, batch_modifications) 
        for instance in batch:
            instance.index_fields(model_bundle.vocab)      
        #####

        # train
        model.train()
        batch = data_loader.collate_fn(batch)
        if data_loader.cuda_device is not None:
            batch = move_to_device(batch, data_loader.cuda_device)

        # forward, backward pass
        optimizer.zero_grad()
        batch.pop('metadata', None)
        batch_outputs = model(**batch)# Dict[str, torch.Tensor]
        batch_loss = batch_outputs["loss"]
        batch_loss.backward()
        train_loss += batch_loss.item()
        optimizer.step()

        #metric
        metrics = model.get_metrics(reset=True)
        metrics["batch_loss"] = batch_loss
        
        batches_in_epoch_completed += 1
        metrics["loss"] = float(train_loss / batches_in_epoch_completed) if batches_in_epoch_completed > 0 else 0.0
        description = training_util.description_from_metrics(metrics)

        print("\n Batch:", batches_in_epoch_completed, " trained.")
        print(description)


    return all_modifications, modifications_history # test accuracy always increase in this epoch

"""
This training function is only used for the testing purpose.
We would use AllenNLP GradientDescent Trainer for evaluation.
=============================================================
"""
def train_epoch(model, data_loader, test_data_loader, optimizer, eval_interval=10) -> Dict[str, float]:
    """
    Trains one epoch
    """
    model_bundle = resource.AllenNLPModelBundle(model, model.vocab)
    train_loss = 0.0
    model.train()
    train_instances = list(data_loader.iter_instances())
    assert len(train_instances) <10000

    num_training_batches = len(data_loader)
    print(f'# of training batches : {str(num_training_batches)} ')
    batches_in_epoch_completed = 0
    list_of_batch_indices = list(data_loader.batch_sampler.get_batch_indices(train_instances))
    metric_history = list()
    for i, dataset_indices in enumerate(list_of_batch_indices):
        if batches_in_epoch_completed % eval_interval == 0:
            test_metric = model_bundle.evaluate_from_dataloader(test_data_loader)['accuracy']
            metric_history.append(test_metric)
        batch = [train_instances[i] for i in dataset_indices]
        # train
        model.train()
        batch = data_loader.collate_fn(batch)
        if data_loader.cuda_device is not None:
            batch = move_to_device(batch, data_loader.cuda_device)

        # forward, backward pass
        optimizer.zero_grad()
        batch_outputs = model(**batch)# Dict[str, torch.Tensor]
        batch_loss = batch_outputs["loss"]
        batch_loss.backward()
        train_loss += batch_loss.item()
        optimizer.step()

        #metric
        metrics = model.get_metrics(reset=True)
        metrics["batch_loss"] = batch_loss
        
        batches_in_epoch_completed += 1
        metrics["loss"] = float(train_loss / batches_in_epoch_completed) if batches_in_epoch_completed > 0 else 0.0
        description = training_util.description_from_metrics(metrics)
        

        print("\n Batch:", batches_in_epoch_completed, " trained.")
        print(description)

    return metric_history

if __name__=="__main__":

    # import my allennlp plugin for unlearnable
    for package_name in include_package:
        import_module_and_submodules(package_name)

    # construct model and dataloader
    params = Params.from_file(param_path) # parse JSON file
    common_util.prepare_environment(params)  # set random seed

    required_params = {k: params.params[k] for k in ['dataset_reader', 'validation_dataset_reader', 'train_data_path', 'test_data_path', 'model', 'data_loader', 'validation_data_loader']}
    object_constructor = utils.GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=serialization_dir)
    model = object_constructor.model
    model = model.cuda(cuda_device)
    data_loader = object_constructor.data_loader
    test_data_loader = object_constructor.test_data_loader
    assert data_loader.batch_sampler is not None # need it for fetching original ids to modify

    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(model_parameters=parameters)
    field_to_modify = "tokens"
    if task == "squad":
        optimizer = AdamOptimizer(model_parameters=parameters, betas=[0.9, 0.9])
        field_to_modify = 'passage' # `generate_modifications` needs it for finding textfield to modify and identifying invalid positions to modify

    # to cuda
    cuda_device=int_to_device(cuda_device)
    data_loader.set_target_device(cuda_device)
    test_data_loader.set_target_device(cuda_device)
    model = model.cuda(cuda_device)
        
    batches_in_epoch_completed = 0        
    
    if mode=="min_min":
        # prepare
        if task=="sst2":
            constraints = [WordEmbeddingDistance(min_cos_sim=0.5), PartOfSpeech()]
        else:
            constraints = []
        model_bundle = resource.AllenNLPModelBundle(model, model.vocab)
        mod_generator = utils.GradientBasedGenerator(model_bundle, constraints=constraints), 
        mod_applicator = utils.ModificationApplicator(type=task)

        # generate
        start = timeit.default_timer()
        all_modifications, modifications_history = generate_mod(model_bundle, data_loader, test_data_loader, optimizer, mod_generator, mod_applicator, num_train_steps_per_perturbation, task=task)
        stop = timeit.default_timer()
        print('############ Time: ', stop - start, ' ############') 
        with open(f'{serialization_dir}/modifications.pickle', 'wb') as file:
            pickle.dump(all_modifications, file)
        with open(f'{serialization_dir}/modifications_history.pickle', 'wb') as file:
            pickle.dump( modifications_history, file)
    elif mode=="train":
        metric_history_across_epochs = list()
        for epoch in range(num_epochs):
            metric_history = train_epoch(model, data_loader, test_data_loader, optimizer)
            metric_history_across_epochs.append(metric_history)
        print(metric_history_across_epochs)
        with open(f'{serialization_dir}/metric_history.pickle', 'wb') as file:
            pickle.dump( metric_history_across_epochs, file )
    
        

