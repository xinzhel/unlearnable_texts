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
from typing import Any
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from utils import TextModifier

logger = logging.getLogger(__name__)

task="sst2"
model_name="lstm"
serialization_dir = f'outputs/{task}/{model_name}'
class_wise = False
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

    parser.add_argument(
        "--class-wise",
        action="store_true",
        default=class_wise,
        help="generaate class-wise triggers",
    )

    # Now we can parse the arguments.
    args = parser.parse_args()

    return parser, args


class GenerateAdversarial(Registrable):

    default_implementation = "default"
    """
    The default implementation is registered as 'default'.
    """

    def __init__(
        self,
        trainer: Trainer = None,
        text_modifier: TextModifier = None,
    ):
        self.trainer = trainer
        self.text_modifier = text_modifier

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: str,
        class_wise: bool,
        dataset_reader: DatasetReader,
        train_data_path: Any,
        model: Lazy[Model],
        data_loader: Lazy[DataLoader],
        trainer: Lazy[Trainer],
        text_modifier: Lazy[TextModifier],
        vocabulary: Lazy[Vocabulary] = Lazy(Vocabulary),
        validation_dataset_reader: DatasetReader = None,
        validation_data_path: Any = None,
        validation_data_loader: Lazy[DataLoader] = None,
    ):
        
        # Train data loader.
        data_loaders = { "train": data_loader.construct(reader=dataset_reader, data_path=train_data_path)}

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

        # construct vocabulary
        instance_generator = (
            instance
            for key, data_loader in data_loaders.items()
            for instance in data_loader.iter_instances()
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)

        # construct vocabulary from transformers (PretrainedTokenizer)
        for field in next(data_loaders['train'].iter_instances()).fields.values():
            from allennlp.data.fields import TextField
            from allennlp.data.token_indexers import PretrainedTransformerIndexer
            if type(field) == TextField:
                for indexer in  field._token_indexers.values():
                    if type(indexer) == PretrainedTransformerIndexer:
                        indexer._add_encoding_to_vocabulary_if_needed(vocabulary_)

        # model
        from allennlp.predictors import Predictor
        predictor = Predictor.from_path(f'models/{task}/{model_name}')
        model = predictor._model

        # Initializing the model can have side effect of expanding the vocabulary.
        vocabulary_path = os.path.join(serialization_dir, "vocabulary")
        vocabulary_.save_to_files(vocabulary_path)

        # indexing
        for data_loader_ in data_loaders.values():
            data_loader_.index_with(model_.vocab)

        # complete its required arguments from command line
        text_modifier = text_modifier.construct(model=model_, data_loader=data_loaders["train"], serialization_dir=serialization_dir, class_wise=class_wise,)

        
        from allennlp.predictors import Predictor
        predictor = Predictor.from_path(f'models/{task}/{model_name}')
        model = predictor._model
        
        return cls(text_modifier=text_modifier)


# TextModifier(model, data_loaders["train"], 
            #                     serialization_dir,
            #                     perturb_bsz,
            #                     cos_sim_constraint=trainer.cos_sim_constraint,
            #                     input_embedder_name=trainer.input_embedder_name,
            #                     input_field_name=trainer.input_field_name,
            #                     map_swap=trainer.map_swap,
            #                     class_wise=class_wise,
            #                     
            #                     )
        


GenerateAdversarial.register("default", constructor="from_partial_objects")(GenerateAdversarial)



if __name__=="__main__":
    # parse command line arguments
    parser, args = parse_cmd_args()

    # import my allennlp plugin for unlearnable
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)

    # parse JSON file
    parameter_filename=args.param_path
    params = Params.from_file(parameter_filename)
    train_loop = GenerateAdversarial.from_params(
        params=params,
        serialization_dir=args.serialization_dir,
        class_wise=args.class_wise,
    )
    
    common_util.prepare_environment(params)

    # generate unlneanable during training
    metrics = train_loop.run()