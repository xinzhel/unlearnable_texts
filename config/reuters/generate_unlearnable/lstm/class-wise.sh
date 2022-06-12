#!/bin/bash

# command line arguments
cd ../../../../
task=sst2
config=lstm
generate_config_path=config/${task}/generate_unlearnable/${config}/${config}.json
serialization_dir=outputs/${task}/${config}

# remove previously-runned information
rm -rf ${serialization_dir}
echo REMOVE ${serialization_dir}

# generate unlearnable
python -u generate_unlearnable.py                                         \
                       --param_path $generate_config_path                 \
                       --serialization-dir $serialization_dir             \
                       --include-package allennlp_extra                   \
                       --class-wise

