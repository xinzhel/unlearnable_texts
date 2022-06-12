import argparse
import os
import logging
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.common.util import import_module_and_submodules
from allennlp.common import Params, Registrable, Lazy
from allennlp.common import util as common_util
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import DataLoader
from allennlp.models.model import Model
from allennlp.training.trainer import Trainer
from typing import Any, Dict, List, Optional, Text, Union
from allennlp_models.classification import *
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from allennlp.common.util import int_to_device
from allennlp.training import util as training_util
import torch
from copy import deepcopy
from allennlp.nn.util import move_to_device
from utils import TextModifier
from allennlp.training.optimizers import AdamOptimizer

logger = logging.getLogger(__name__)

task="sst2"
model_name="lstm"
serialization_dir = f'outputs/{task}/{model_name}'
cuda_device = 0
num_epochs=1
num_train_steps_per_perturbation=30
def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--param_path", 
        type=str,
        default=f'config/{task}/{model_name}.jsonnet',
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

"""
Construct Model and DataLoader
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
        data_loader,
    ):
        self.model = model 
        self.data_loader = data_loader

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        dataset_reader: DatasetReader,
        train_data_path: Any,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
    ):
        
        # model
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        vocabulary = Vocabulary.from_files(directory=vocab_dir)
        model_ = model.construct(vocab=vocabulary)

        # data loader
        data_loader = data_loader.construct(reader=dataset_reader, data_path=train_data_path)
        data_loader.index_with(model_.vocab)

        return cls( model=model_, data_loader=data_loader )     

GenerateUnlearnable.register("default", constructor="from_partial_objects")(GenerateUnlearnable)


"""
train 
============================================
"""
def train_epoch(model, data_loader, optimizer, text_modifier, num_train_steps_per_perturbation) -> Dict[str, float]:
    """
    Trains one epoch
    """

    train_loss = 0.0
    model.train()
    train_instances = list(data_loader.iter_instances())

    num_training_batches = len(data_loader)
    print(f'# of training batches : {str(num_training_batches)} ')
    batches_in_epoch_completed = 0

    for batch_indices in data_loader.batch_sampler.get_batch_indices(train_instances):
        
        batch = [train_instances[i] for i in batch_indices]
        # Set the model to "train" mode.
        model.train()
        if batches_in_epoch_completed > num_train_steps_per_perturbation:  
            # apply unleanrable modifications
            batch = text_modifier.modify(deepcopy(batch), batch_indices)

        # instances to tensor_dict: originally done in data_loader
        batch = data_loader.collate_fn(batch)
        if data_loader.cuda_device is not None:
            batch = move_to_device(batch, data_loader.cuda_device)

        # forward , backward pass
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

        if batches_in_epoch_completed % num_train_steps_per_perturbation == 0:
            text_modifier.update(epoch, batches_in_epoch_completed)
        

if __name__=="__main__":
    # parse command line arguments
    parser, args = parse_cmd_args()

    # import my allennlp plugin for unlearnable
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)

    # construct model and dataloader
    params = Params.from_file(args.param_path) # parse JSON file
    common_util.prepare_environment(params)  # set random seed
    required_params = {k: params.params[k] for k in ['dataset_reader', 'train_data_path', 'model', 'data_loader']}
    object_constructor = GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=serialization_dir)
    model = object_constructor.model
    model = model.cuda(cuda_device)
    data_loader = object_constructor.data_loader
    assert data_loader.batch_sampler is not None # need it for fetching original ids to modify

    # generate text modifications
    text_modifier_ = TextModifier(
        model=model, 
        data_loader=object_constructor.data_loader,
        serialization_dir=args.serialization_dir, 
        class_wise=False,
        only_where_to_modify=False,
        max_swap=1,
        perturb_bsz=32,
        constraints=[],
        input_field_name="tokens"
    )

    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(model_parameters=parameters)

    # to cuda
    cuda_device=int_to_device(cuda_device)
    data_loader.set_target_device(cuda_device)
    model = model.cuda(cuda_device)
        
    batches_in_epoch_completed = 0        
        
    for epoch in range(num_epochs):
        train_epoch(epoch)

