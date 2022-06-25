import pickle
import argparse
from allennlp.common.util import import_module_and_submodules
from allennlp.common import Params
from allennlp.common import util as common_util
import utils
import pandas as pd


def parse_args(task="sst2", model_name="lstm", ):
    # from https://github.com/allenai/allennlp/blob/5338bd8b4a7492e003528fe607210d2acc2219f5/allennlp/commands/train.py
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", 
        type=str,
        default=task,
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        default=model_name,
    )

    parser.add_argument(
        "-s",
        "--serialization_dir",
        type=str,
        default=f'outputs/{task}/{model_name}',
        help="directory in which to save the model and its logs",
    )

    parser.add_argument(
        "--mod_file_name",
        type=str,
        default="modifications_30"
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


if __name__=="__main__":
    _, args = parse_args()
    
    param_path = f'config/{args.task}/{args.model_name}.jsonnet'
    # import my allennlp plugin for unlearnable
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    # construct model and dataloader
    params = Params.from_file(param_path) # parse JSON file
    common_util.prepare_environment(params)  # set random seed
   
    required_params = {k: params.params[k] for k in ['dataset_reader', 'train_data_path', 'data_loader']}
    object_constructor = utils.GenerateUnlearnable.from_params(params=Params(params=required_params),serialization_dir=args.serialization_dir)
    data_loader = object_constructor.data_loader
    with open(f'{args.serialization_dir}/{args.mod_file_name}.pickle', 'rb') as file:
        modifications = pickle.load(file)
    mod_applicator = utils.ModificationApplicator(type=args.task)

    train_instances = list(data_loader.iter_instances())
    modified_instances = mod_applicator.apply_modifications(train_instances, modifications)

    instances_dict = []
    for inst in  modified_instances:
        instances_dict.append({"label": int(inst.fields['label'].label), "text": ' '.join(inst.fields['tokens'].human_readable_repr())})
    df = pd.DataFrame(instances_dict)
    df.to_json(f"{args.serialization_dir}/train_{args.mod_file_name}.json", orient = "records", lines=True)