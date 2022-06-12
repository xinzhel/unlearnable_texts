#!/bin/bash

# command line arguments
cd ../../../
task=ag_news
for model_name in attention lstm cnn
do
    generate_config_path=config/${task}/generate_unlearnable/${model_name}.jsonnet
    serialization_dir=outputs/${task}/${model_name}

    # remove previously-runned information
    rm -rf ${serialization_dir}
    echo REMOVE ${serialization_dir}

    # generate unlearnable
    python -u generate_unlearnable.py                                         \
                        --param_path $generate_config_path                 \
                        --serialization-dir $serialization_dir             \
                        --include-package allennlp_extra                   \
                        --class-wise

done