import logging
import os
from typing import Optional, Dict, Any, List, Union, Tuple, TYPE_CHECKING

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.training.callbacks.log_writer import LogWriterCallback


if TYPE_CHECKING:
    from allennlp.training.gradient_descent_trainer import GradientDescentTrainer


logger = logging.getLogger(__name__)



@TrainerCallback.register("my_wandb")
class WandBCallback(LogWriterCallback):
    """ A copy from `allennlp.training.WandBCallback` with the following modifications
    1. not read allennlp configuration file for wandb config
    """

    def __init__(
        self,
        serialization_dir: str,
        summary_interval: int = 100,
        distribution_interval: Optional[int] = None,
        batch_size_interval: Optional[int] = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        watch_model: bool = True,
        files_to_save: Tuple[str, ...] = ("config.json", "out.log"),
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if "WANDB_API_KEY" not in os.environ:
            logger.warning(
                "Missing environment variable 'WANDB_API_KEY' required to authenticate to Weights & Biases."
            )

        super().__init__(
            serialization_dir,
            summary_interval=summary_interval,
            distribution_interval=distribution_interval,
            batch_size_interval=batch_size_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
        )

        self._watch_model = watch_model
        self._files_to_save = files_to_save
        self._run_id: Optional[str] = None
        self._wandb_kwargs: Dict[str, Any] = dict(
            dir=os.path.abspath(serialization_dir),
            project=project,
            entity=entity,
            group=group,
            name=name,
            notes=notes,
            config={},
            tags=tags,
            anonymous="allow",
            **(wandb_kwargs or {}),
        )

    @overrides
    def log_scalars(
        self,
        scalars: Dict[str, Union[int, float]],
        log_prefix: str = "",
        epoch: Optional[int] = None,
    ) -> None:
        self._log(scalars, log_prefix=log_prefix, epoch=epoch)

    @overrides
    def log_tensors(
        self, tensors: Dict[str, torch.Tensor], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        self._log(
            {k: self.wandb.Histogram(v.cpu().data.numpy().flatten()) for k, v in tensors.items()},  # type: ignore
            log_prefix=log_prefix,
            epoch=epoch,
        )

    def _log(
        self, dict_to_log: Dict[str, Any], log_prefix: str = "", epoch: Optional[int] = None
    ) -> None:
        if log_prefix:
            dict_to_log = {f"{log_prefix}/{k}": v for k, v in dict_to_log.items()}
        if epoch is not None:
            dict_to_log["epoch"] = epoch
        self.wandb.log(dict_to_log, step=self.trainer._total_batches_completed)  # type: ignore

    @overrides
    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        super().on_start(trainer, is_primary=is_primary, **kwargs)

        if not is_primary:
            return None

        import wandb

        self.wandb = wandb

        if self._run_id is None:
            self._run_id = self.wandb.util.generate_id()

        self.wandb.init(id=self._run_id, **self._wandb_kwargs)

        for fpath in self._files_to_save:
            self.wandb.save(  # type: ignore
                os.path.join(self.serialization_dir, fpath), base_path=self.serialization_dir
            )

        if self._watch_model:
            self.wandb.watch(self.trainer.model)  # type: ignore

    @overrides
    def close(self) -> None:
        super().close()
        self.wandb.finish()  # type: ignore

    def state_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._wandb_kwargs["resume"] = "auto"
        self._run_id = state_dict["run_id"]
