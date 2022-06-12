#!/bin/bash

# command line arguments
cd ../../../../
task='sst2'
model_type='cnn'
train_config_path=config/${task}/apply_unlearnable/${model_type}/${model_type}.json
serialization_dir=models/${task}/${model_type}

# remove previously-runed information
rm -rf ${serialization_dir}


# apply unlearnable and train
python -u apply_unlearnable.py      $train_config_path              \
                    --serialization-dir $serialization_dir          \
                    --include-package allennlp_extra
