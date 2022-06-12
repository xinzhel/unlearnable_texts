#!/bin/bash

# command line arguments
cd ../../../../
task='sst2'
model_type='bert' #'lstm', 'cnn'
train_config_path=config/${task}/apply_unlearnable/${model_type}.jsonnet
serialization_dir=models/${task}/${model_type}
echo ${train_config_path}
# remove previously-runed information
rm -rf ${serialization_dir}


# apply unlearnable and train
python apply_unlearnable.py                                     \
                    --param-path $train_config_path             \
                    --serialization-dir $serialization_dir      \
                    --include-package allennlp_extra
