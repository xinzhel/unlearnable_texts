#!/bin/bash

# command line arguments
cd ../../../
task='ag_news'
for model_type in attention lstm cnn bert
do
    train_config_path=config/${task}/apply_unlearnable//${model_type}.jsonnet
    serialization_dir=models/${task}/${model_type}

    # remove previously-runed information
    rm -rf ${serialization_dir}


    # apply unlearnable and train
    python -u apply_unlearnable.py                                   \
                        --param-path $train_config_path              \
                        --serialization-dir $serialization_dir       \
                        --include-package allennlp_extra
done