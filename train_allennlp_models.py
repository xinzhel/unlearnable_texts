import argparse
import logging
import os
import shutil
import warnings
from os import PathLike
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from allennlp.common import Lazy, Params, Registrable
from allennlp.common import logging as common_logging
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.models.archival import (CONFIG_NAME, archive_model,
                                      verify_include_in_archive)
from allennlp.models.model import _DEFAULT_WEIGHTS, Model
from allennlp.training import util as training_util
from allennlp.training.trainer import Trainer
from allennlp_models.generation import *
from allennlp_models.classification import *
from allennlp_models.pair_classification.dataset_readers import *
from allennlp_models.rc import *
from utils import PerturbedTransformerSquadReader, PerturbLabeledTextDatasetReader


task="squad"
model_name="bidaf_glove"
serialization_dir = f'../models/{task}/'
modified_train_path = None#f"outputs/{task}/lstm/train_modifications_30.json"
cuda_device = 0
recover = False
force = True

def parse_train_args():
    # from https://github.com/allenai/allennlp/blob/5338bd8b4a7492e003528fe607210d2acc2219f5/allennlp/commands/train.py
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", 
        type=str,
        default=None,
        help="path to parameter file describing the model to be trained"
    )
    parser.add_argument(
        "--model-name", 
        type=str,
        help="path to parameter file describing the model to be trained"
    )

    parser.add_argument(
        "-s",
        "--serialization-dir",
        type=str,
        default=serialization_dir,
        help="directory in which to save the model and its logs",
    )

    parser.add_argument(
        "--modified-train-path", 
        type=str, 
        default= f'config/{task}/{model_name}.jsonnet',
        # 'config/snli/apply_unlearnable/esim/esim.jsonnet',
        help="path to parameter file describing the model to be trained"
    )

    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        default=recover,
        help="recover training from the state in serialization_dir",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=force,
        help="overwrite the output directory if it exists",
    )

    parser.add_argument(
        "-o",
        "--overrides",
        type=str,
        default="",
        help=(
            "a json(net) structure used to override the experiment configuration, e.g., "
            "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
            " with nested dictionaries or with dot syntax."
        ),
    )

    parser.add_argument(
        "--node_rank", type=int, default=0, help="rank of this node in the distributed setup"
    )

    parser.add_argument(
        "--file-friendly-logging",
        action="store_true",
        default=False,
        help="outputs tqdm status on separate lines and slows tqdm refresh rate",
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


class MyTrainModel(Registrable):
    """
    A copy of `commands.train.TrainModel` with the extra functions:
    1. add test_data_loader into trainer
    """

    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
        self,
        serialization_dir: str,
        model: Model,
        trainer: Trainer,
        evaluation_data_loader: DataLoader = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
         **kwargs,
    ) -> None:
        self.serialization_dir = serialization_dir
        self.model = model
        self.trainer = trainer
        self.evaluation_data_loader = evaluation_data_loader
        self.evaluate_on_test = evaluate_on_test
        self.batch_weight_key = batch_weight_key

    def run(self) -> Dict[str, Any]:
        return self.trainer.train()

    def finish(self, metrics: Dict[str, Any]):
#         if self.evaluation_data_loader is not None and self.evaluate_on_test:
#             logger.info("The model will be evaluated using the best epoch weights.")
#             test_metrics = training_util.evaluate(
#                 self.model,
#                 self.evaluation_data_loader,
#                 cuda_device=self.trainer.cuda_device,
#                 batch_weight_key=self.batch_weight_key,
#             )
# 
#             for key, value in test_metrics.items():
#                 metrics["test_" + key] = value
#         elif self.evaluation_data_loader is not None:
#             logger.info(
#                 "To evaluate on the test set after training, pass the "
#                 "'evaluate_on_test' flag, or use the 'allennlp evaluate' command."
#             )
        common_util.dump_metrics(
            os.path.join(self.serialization_dir, "metrics.json"), metrics, log=True
        )

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        local_rank: int,
        dataset_reader: DatasetReader,
        train_data_path: Any,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
        datasets_for_vocab_creation: List[str] = None,
        validation_dataset_reader: DatasetReader = None,
        validation_data_path: Any = None,
        validation_data_loader: Lazy[DataLoader] = None,
        test_data_path: Any = None,
        evaluate_on_test: bool = False,
        batch_weight_key: str = "",
         **kwargs,
    ) -> "MyTrainModel":

        # Train data loader.
        data_loaders: Dict[str, DataLoader] = {
            "train": data_loader.construct(reader=dataset_reader, data_path=train_data_path)
        }

        # Validation data loader.
        if validation_data_path is not None:
            validation_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["validation"] = validation_data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
            else:
                data_loaders["validation"] = data_loader.construct(
                    reader=validation_dataset_reader, data_path=validation_data_path
                )
                if getattr(data_loaders["validation"], "batches_per_epoch", None) is not None:
                    warnings.warn(
                        "Using 'data_loader' params to construct validation data loader since "
                        "'validation_data_loader' params not specified, but you have "
                        "'data_loader.batches_per_epoch' set which may result in different "
                        "validation datasets for each epoch.",
                        UserWarning,
                    )

        # Test data loader.
        if test_data_path is not None:
            test_dataset_reader = validation_dataset_reader or dataset_reader
            if validation_data_loader is not None:
                data_loaders["test"] = validation_data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )
            else:
                data_loaders["test"] = data_loader.construct(
                    reader=test_dataset_reader, data_path=test_data_path
                )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in data_loaders:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

        instance_generator = (
            instance
            for key, data_loader in data_loaders.items()
            if datasets_for_vocab_creation is None or key in datasets_for_vocab_creation
            for instance in data_loader.iter_instances()
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)

        # construct vocabulary from transformers (PretrainedTokenizer)
        for field in next(data_loaders['train'].iter_instances()).fields.values():
            from allennlp.data.fields import TextField
            from allennlp.data.token_indexers import \
                PretrainedTransformerIndexer
            if type(field) == TextField:
                for indexer in  field._token_indexers.values():
                    if type(indexer) == PretrainedTransformerIndexer:
                        indexer._add_encoding_to_vocabulary_if_needed(vocabulary_)

        model_ = model.construct(vocab=vocabulary_, serialization_dir=serialization_dir)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the primary. In the degenerate non-distributed
        # case, we're trivially the primary. In the distributed case this is safe
        # to do without worrying about race conditions since saving and loading
        # the vocab involves acquiring a file lock.
        if local_rank == 0:
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            vocabulary_.save_to_files(vocabulary_path)

        for data_loader_ in data_loaders.values():
            data_loader_.index_with(model_.vocab)

         
        trainer_ = trainer.construct(
            serialization_dir=serialization_dir,
            model=model_,
            data_loader=data_loaders["train"],
            validation_data_loader=data_loaders.get("validation"),
            test_data_loader=data_loaders["test"] if 'test' in data_loaders.keys() else None,
            local_rank=local_rank,
        )
        assert trainer_ is not None

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=data_loaders.get("test"),
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )


MyTrainModel.register("default", constructor="from_partial_objects")(MyTrainModel) 

def _train_worker(
    process_rank: int,
    params: Params,
    serialization_dir: Union[str, PathLike],
    include_package: List[str] = None,
    dry_run: bool = False,
    node_rank: int = 0,
    primary_addr: str = "127.0.0.1",
    primary_port: int = 29500,
    world_size: int = 1,
    distributed_device_ids: List[int] = None,
    file_friendly_logging: bool = False,
    include_in_archive: List[str] = None,
) -> Optional[Model]:
    
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    common_logging.prepare_global_logging(
        serialization_dir,
        rank=process_rank,
        world_size=world_size,
    )
    common_util.prepare_environment(params)

    distributed = world_size > 1

    primary = process_rank == 0

    include_package = include_package or []

    if distributed:
        assert distributed_device_ids is not None

        # Since the worker is spawned and not forked, the extra imports need to be done again.
        # Both the ones from the plugins and the ones from `include_package`.
        import_plugins()
        for package_name in include_package:
            common_util.import_module_and_submodules(package_name)

        num_procs_per_node = len(distributed_device_ids)
        # The Unique identifier of the worker process among all the processes in the
        # distributed training group is computed here. This is used while initializing
        # the process group using `init_process_group`
        global_rank = node_rank * num_procs_per_node + process_rank

        # Number of processes per node is useful to know if a process
        # is a primary in the local node(node in which it is running)
        os.environ["ALLENNLP_PROCS_PER_NODE"] = str(num_procs_per_node)

        # In distributed training, the configured device is always going to be a list.
        # The corresponding gpu id for the particular worker is obtained by picking the id
        # from the device list with the rank as index
        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        # Till now, "cuda_device" might not be set in the trainer params.
        # But a worker trainer needs to only know about its specific GPU id.
        params["trainer"]["cuda_device"] = gpu_id
        params["trainer"]["world_size"] = world_size
        params["trainer"]["distributed"] = True

        if gpu_id >= 0:
            torch.cuda.set_device(int(gpu_id))
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{primary_addr}:{primary_port}",
                world_size=world_size,
                rank=global_rank,
            )
        logging.info(
            f"Process group of world size {world_size} initialized "
            f"for distributed training in worker {global_rank}"
        )

    train_loop = MyTrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=process_rank,
    )

    if dry_run:
        return
    try:
        if distributed:  # let the setup get ready for all the workers
            dist.barrier()

        metrics = train_loop.run()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if primary and os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            best_weights_path = train_loop.trainer.get_best_weights_path()
            if best_weights_path is None:
                logging.info(
                    "Training interrupted by the user, and no best model has been saved. "
                    "No model archive created."
                )
            else:
                logging.info(
                    "Training interrupted by the user. Attempting to create "
                    "a model archive using the current best epoch weights."
                )
                archive_model(
                    serialization_dir,
                    weights=best_weights_path,
                    include_in_archive=include_in_archive,
                )
        raise

    if primary:
        train_loop.finish(metrics)

    if not distributed:
        return train_loop.model

    return None

def main():
#     train_data_path = "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt"
#     clean_data_reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class")
#     instances = list(clean_data_reader.read(train_data_path))
#     modifications_path = 'outputs/sst2/lstm/modification_epoch0_batch180.json'
# 
#     perturbed_reader = PerturbedSSTDatasetReader(modification_path=modifications_path, granularity="2-class")
#     modifications = perturbed_reader.modifications
#  instance_generator = iter(perturbed_reader.read(train_data_path))
    parser, args = parse_train_args()
    if task is None:
        task = args.task
        model_name = args.model_name
        param_path = f'config/{task}/{model_name}.jsonnet'
        modified_train_path = args.modified_train_path
    
    # Import any additional modules needed (to register custom classes).
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)

    params = Params.from_file(param_path, args.overrides)
    if cuda_device:
        params['trainer']['cuda_device'] = cuda_device
    if modified_train_path:
        params['train_data_path'] = modified_train_path

    # create serialization dir
    serialization_dir=args.serialization_dir
    recover=args.recover
    force=args.force
    node_rank=args.node_rank
    include_package=args.include_package
    training_util.create_serialization_dir(params, serialization_dir, recover, force)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME)) # wandb callback needs config.json in this dir

    # extra archive
    include_in_archive = params.pop("include_in_archive", None)
    verify_include_in_archive(include_in_archive)

    # logging
    common_logging.prepare_global_logging(
        serialization_dir,
        rank=0,
        world_size=1,
    )

    # train: running multiple processes for training or not
    distributed_params = params.params.pop("distributed", None)
    if distributed_params is None:
        
        # random seeds
        common_util.prepare_environment(params)
        train_loop = MyTrainModel.from_params(
            params=params,
            serialization_dir=serialization_dir,
            local_rank=0,
        )
        # train
        metrics = train_loop.run()
        train_loop.finish(metrics)

    else:
        # We are careful here so that we can raise a good error if someone
        # passed the wrong thing - cuda_devices are required.
        device_ids = distributed_params.pop("cuda_devices", None)
        multi_device = isinstance( device_ids, list ) and len(device_ids) > 1
        num_nodes = distributed_params.pop("num_nodes", 1)

        if not (multi_device or num_nodes > 1):
            raise ConfigurationError(
                "Multiple cuda devices/nodes need to be configured to run distributed training."
            )
        check_for_gpu(device_ids)

        primary_addr = "127.0.0.1" # running locally
        # we can automatically find an open port if one is not specified.
        primary_port = (
            distributed_params.pop("primary_port", None) or common_util.find_open_port()
        )

        num_procs = len(device_ids)
        world_size = num_nodes * num_procs

        # Creating `Vocabulary` objects from workers could be problematic since
        # the data loaders in each worker will yield only `rank` specific
        # instances. Hence it is safe to construct the vocabulary and write it
        # to disk before initializing the distributed context. The workers will
        # load the vocabulary from the path specified.
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        if recover:
            vocab = Vocabulary.from_files(vocab_dir)
        else:
            vocab = training_util.make_vocab_from_params(
                params.duplicate(), serialization_dir, print_statistics=False
            )
        params["vocabulary"] = {
            "type": "from_files",
            "directory": vocab_dir,
            "padding_token": vocab._padding_token,
            "oov_token": vocab._oov_token,
        }

        logging.info(
            "Switching to distributed training mode since multiple GPUs are configured | "
            f"Primary is at: {primary_addr}:{primary_port} | Rank of this node: {node_rank} | "
            f"Number of workers in this node: {num_procs} | Number of nodes: {num_nodes} | "
            f"World size: {world_size}"
        )

        mp.spawn(
            _train_worker,
            args=(
                params.duplicate(),
                serialization_dir,
                include_package,
                False,
                node_rank,
                primary_addr,
                primary_port,
                world_size,
                device_ids,
                False,
                include_in_archive,
            ),
            nprocs=num_procs,
        )
    # archive_model(serialization_dir, include_in_archive=include_in_archive)
    # model = Model.load(params, serialization_dir)
            

if __name__=="__main__":
    main()

# ========
# debug.py
# ========
# import sys
# import shutil
# from train_allennlp_models import main
# 
# task = "squad"
# config = 'transformer_qa'
# 
# config_file = f"config/{task}/apply_unlearnable/{config}/{config}.jsonnet"
# serialization_dir = f"models/{task}/{config}"
# shutil.rmtree(serialization_dir, ignore_errors=True)
# sys.argv = [
#     "python",  # useless since we directly call `main()``
#     config_file,
#     "--serialization-dir", serialization_dir,
#     "--include-package", "allennlp_extra",
# ]
# 
# main()
    
    

  

   
