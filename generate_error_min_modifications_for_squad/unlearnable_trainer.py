import datetime
import glob
import logging
import math
import os
import re
import time
import pickle
import warnings
from typing import Iterator, Optional, Union, List, Dict, Tuple, Any, Type, Iterable
from copy import deepcopy

import torch
from torch.cuda import amp
import numpy as np
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch import backends

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util, Tqdm, Lazy
from allennlp.nn.util import dist_reduce_sum, find_embedding_layer, move_to_device
from allennlp.data.data_loaders.data_loader import DataLoader, TensorDict
from allennlp.data import Batch, DatasetReader, Instance, Token
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.models.model import Model
from allennlp.modules import token_embedders
from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import Trainer, TrainerCheckpoint
from allennlp.training import util as training_util
from allennlp_extra.text_modifier import TextModifier, prepend_batch

logger = logging.getLogger(__name__)


@Trainer.register("unlearnable", constructor="from_partial_objects")
class UnlearnableTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        text_modifier: TextModifier = None,
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[Union[str, os.PathLike]] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        num_train_steps_per_perturbation: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
        )

        self.model = model
        self.data_loader = data_loader
        assert self.data_loader.batch_sampler is not None
        self.data_loader.set_target_device(self.cuda_device)
        if cuda_device >=0:
            model = model.cuda(self.cuda_device)
        self._validation_data_loader = validation_data_loader
        if self._validation_data_loader is not None:
            self._validation_data_loader.set_target_device(self.cuda_device)
        self.optimizer = optimizer

        # self._metric_tracker = MetricTracker(validation_metric, patience)
        self._num_epochs = num_epochs
        self._batches_in_epoch_completed = 0        
        
        # unlearnable
        self.num_train_steps_per_perturbation = num_train_steps_per_perturbation
        self.text_modifier = text_modifier
            

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)

        train_loss = 0.0
        self.model.train()
        train_instances = list(self.data_loader.iter_instances())

        num_training_batches = len(self.data_loader)
        logger.info(f'# of training batches : {str(num_training_batches)} ')

        for batch_indices in self.data_loader.batch_sampler.get_batch_indices(train_instances):
            
            batch = [train_instances[i] for i in batch_indices]
            # Set the model to "train" mode.
            self.model.train()
            if self._batches_in_epoch_completed > self.num_train_steps_per_perturbation:  
                # apply unleanrable modifications
                batch = self.text_modifier.modify(deepcopy(batch), batch_indices)

            # instances to tensor_dict: originally done in data_loader
            batch = self.data_loader.collate_fn(batch)
            if self.data_loader.cuda_device is not None:
                batch = move_to_device(batch, self.data_loader.cuda_device)

            # forward , backward pass
            self.optimizer.zero_grad()
            batch.pop('metadata', None)
            batch_outputs = self.model(**batch)# Dict[str, torch.Tensor]
            batch_loss = batch_outputs["loss"]
            batch_loss.backward()
            train_loss += batch_loss.item()
            self.optimizer.step()
   
            #metric
            metrics = self.model.get_metrics(reset=True)
            metrics["batch_loss"] = batch_loss
            
            self._batches_in_epoch_completed += 1
            metrics["loss"] = float(train_loss / self._batches_in_epoch_completed) if self._batches_in_epoch_completed > 0 else 0.0
            description = training_util.description_from_metrics(metrics)

            print("\n Batch:", self._batches_in_epoch_completed, " trained.")
            print(description)

            if self._batches_in_epoch_completed % self.num_train_steps_per_perturbation == 0:
                self.text_modifier.update(epoch, self._batches_in_epoch_completed)
                
            # # valid
            # with torch.no_grad():
            #     # We have a validation set, so compute all the metrics on it.
            #     val_loss, num_batches = self._validation_loss()
            #     val_metrics = self.model.get_metrics(reset=True)
            #     val_metrics["loss"] = float(val_loss / num_batches) if num_batches > 0 else 0.0
            #     print(val_metrics)


    def train(self):
        for epoch in range(self._num_epochs):
            self._train_epoch(epoch)
            self._batches_in_epoch_completed = 0


    def _validation_loss(self) -> Tuple[float, Optional[float], int]:
        
        self.model.eval()

        batches_this_epoch = 0
        val_loss = 0.0
        val_batch_loss = 0.0
        
        for batch in self._validation_data_loader:
            batch.pop('metadata', None)
            batch_outputs = self.model(**batch)
            loss = batch_outputs.get("loss")
            if loss is not None:
                batches_this_epoch += 1
                val_batch_loss = loss.item()
                val_loss += val_batch_loss
                
        return val_loss, batches_this_epoch

    # it is import to have `from_partial_objects` to 
    # construct Lazy objects, e.g., optimizer; 
    # BTW,if we set :Optimizer instead of Lazy[Optimizer], it will get the orig dict
    @classmethod
    def from_partial_objects(
        cls,
        model: Model,
        serialization_dir: str,
        data_loader: DataLoader,
        text_modifier: TextModifier = None, 
        validation_data_loader: DataLoader = None,
        local_rank: int = 0,
        patience: int = None,
        validation_metric: Union[str, List[str]] = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Union[float, bool] = False,
        grad_clipping: float = None,
        distributed: bool = False,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        optimizer: Lazy[Optimizer] = Lazy(Optimizer.default),
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Optional[Lazy[Checkpointer]] = Lazy(Checkpointer),
        run_confidence_checks: bool = True,
        **kwargs,
    ) -> Trainer:

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        moving_average_ = (
            None if moving_average is None else moving_average.construct(parameters=parameters)
        )
        learning_rate_scheduler_ = (
            None
            if learning_rate_scheduler is None
            else learning_rate_scheduler.construct(
                optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
            )
        )
        momentum_scheduler_ = (
            None
            if momentum_scheduler is None
            else momentum_scheduler.construct(optimizer=optimizer_)
        )
        checkpointer_ = (
            None
            if checkpointer is None
            else checkpointer.construct(serialization_dir=serialization_dir)
        )


        return cls(
            model,
            optimizer_,
            data_loader,
            text_modifier = text_modifier, 
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            use_amp=use_amp,
            run_confidence_checks=run_confidence_checks,
            **kwargs,
        )

        

