import pickle
import argparse
import os
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
import resource
from textattack.constraints.grammaticality.part_of_speech import PartOfSpeech
from textattack.constraints.semantics import WordEmbeddingDistance


def parse_args(task="sst2", model_name="lstm", ):
    # from https://github.com/allenai/allennlp/blob/5338bd8b4a7492e003528fe607210d2acc2219f5/allennlp/commands/train.py
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", 
        type=str,
        default=task,
        help="path to parameter file describing the model to be trained"
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        default=model_name,
        help="path to parameter file describing the model to be trained"
    )

    parser.add_argument(
        "-s",
        "--serialization_dir",
        type=str,
        default=f'outputs/{task}/{model_name}',
        help="directory in which to save the model and its logs",
    )

    parser.add_argument(
        "--num_train_steps_per_perturbation",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
    )

    parser.add_argument(
                    "--include-package",
                    type=str,
                    action="append",
                    default=['allennlp_extra'],
                    help="additional packages to include",
                )


    # Now we can parse the arguments.
    args = parser.parse_args()
    return parser, args

"""
generate error min modifications 
============================================
"""
from allennlp.data.tokenizers import Token
def generate_mod(model_bundle, data_loader, test_data_loader, optimizer, mod_generator, mod_applicator,  num_train_steps_per_perturbation=30, task=None, perturb=True, report_approx_loss_change=False,) -> Dict[str, float]:
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
    # for instance in train_instances:
    #     if instance.fields['label'].label == '1':
    #         instance.fields['tokens'].tokens += [Token('!')]
    #     else:
    #         instance.fields['tokens'].tokens += [Token('@')]
    
    batches_in_epoch_completed = 0
    current_test_accuracy = None
    # if report_approx_loss_change:
    #     num_batches=60
    #     num_train = data_loader.batch_sampler.batch_size * num_batches
    #     random.seed(22)
    #     random.shuffle(train_instances)
    #     train_instances = train_instances[:num_train]
    #     print(f'only use {num_batches} out of {str(len(data_loader))} total batches')
    assert data_loader.batch_sampler is not None # need it for fetching original ids to modify
    list_of_batch_indices = list(data_loader.batch_sampler.get_batch_indices(train_instances))
    loss1, loss2, loss_change = [], [], []
    for i, dataset_indices in enumerate(list_of_batch_indices):
        if  report_approx_loss_change and i == 0:
            batch_orig = [train_instances[i] for i in dataset_indices]
            l1, l2, lc = utils.verify_loss_change(batch_orig, model_bundle, data_loader)
            loss1.append(l1)
            loss2.append(l2)
            loss_change.append(lc)
            print("#####approximate loss change: ", l1, l2, lc)

        # generate unleanrable modifications
        if perturb and batches_in_epoch_completed % num_train_steps_per_perturbation == 0:
            file_name = f'{args.serialization_dir}/modifications_{batches_in_epoch_completed}.pickle'

            if os.path.exists(file_name):
                with open(file_name, 'rb') as file:
                    all_modifications = pickle.load( file)
            else:
                # get current test accuracy
                test_metric = model_bundle.evaluate_from_dataloader(test_data_loader)[metric]
                if current_test_accuracy is None or test_metric > current_test_accuracy:
                    current_test_accuracy = test_metric
                else:
                    return all_modifications
                
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
                with open(file_name, 'wb') as file:
                    pickle.dump( all_modifications, file)
            
 
        batch = [train_instances[i] for i in dataset_indices]
        ##### apply unleanrable modifications
        if perturb:
            batch_modifications = [all_modifications[i] for i in dataset_indices]
            batch = mod_applicator.apply_modifications(batch, batch_modifications) 
        for instance in batch:
            instance.indexed = False
            instance.index_fields(model_bundle.vocab)      
        #####
        
        # check loss of batch_orig
        # batch_orig = [train_instances[i] for i in dataset_indices]
        # batch_orig = data_loader.collate_fn(batch_orig)
        # if data_loader.cuda_device is not None:
        #     batch_orig = move_to_device(batch_orig, data_loader.cuda_device)
        # batch_outputs = model(**batch_orig)# Dict[str, torch.Tensor]
        # metrics = model.get_metrics(reset=True)
        # batch_loss = batch_outputs["loss"].item()
        # print(batch_loss)

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

        if  report_approx_loss_change and metrics['accuracy']>0.9:
            batch_orig = [train_instances[i] for i in dataset_indices]
            l1, l2, lc = utils.verify_loss_change(batch_orig, model_bundle, data_loader)
            loss1.append(l1)
            loss2.append(l2)
            loss_change.append(lc)
            print("#####approximate loss change: ", l1, l2, lc)

    if perturb:
        return all_modifications # test accuracy always increase in this epoch

if __name__=="__main__":
    _, args = parse_args()
    args.param_path = f'config/{args.task}/{args.model_name}.jsonnet'
    # import my allennlp plugin for unlearnable
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    # construct model and dataloader
    params = Params.from_file(args.param_path) # parse JSON file
    common_util.prepare_environment(params)  # set random seed

    required_params = {k: params.params[k] for k in ['dataset_reader', 'validation_dataset_reader', 'train_data_path', 'test_data_path', 'model', 'data_loader', 'validation_data_loader']}
    object_constructor = utils.GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=args.serialization_dir)
    model = object_constructor.model
    model = model.cuda(args.cuda_device)
    data_loader = object_constructor.data_loader
    test_data_loader = object_constructor.test_data_loader
    

    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(model_parameters=parameters)
    field_to_modify = "tokens"
    if args.task == "squad":
        optimizer = AdamOptimizer(model_parameters=parameters, betas=[0.9, 0.9])
        field_to_modify = 'passage' # `generate_modifications` needs it for finding textfield to modify and identifying invalid positions to modify

    # to cuda
    cuda_device=int_to_device(args.cuda_device)
    data_loader.set_target_device(cuda_device)
    test_data_loader.set_target_device(cuda_device)
    model = model.cuda(cuda_device)
        
    batches_in_epoch_completed = 0        
    
    # prepare
    if args.task=="sst2":
        constraints = [PartOfSpeech(), WordEmbeddingDistance(min_cos_sim=0.5)]
    else:
        constraints = []
    model_bundle = resource.AllenNLPModelBundle(model, model.vocab)
    
    mod_generator = utils.GradientBasedGenerator(model_bundle, constraints=constraints) 
    mod_applicator = utils.ModificationApplicator(type=args.task)

    # generate
    start = timeit.default_timer()
    all_modifications = generate_mod(model_bundle, data_loader, test_data_loader, optimizer, \
        mod_generator, mod_applicator, args.num_train_steps_per_perturbation, task=args.task, \
        perturb=True, report_approx_loss_change=False)
    stop = timeit.default_timer()
    print('############ Time: ', stop - start, ' ############') 
    with open(f'{args.serialization_dir}/modifications.pickle', 'wb') as file:
        pickle.dump(all_modifications, file)
