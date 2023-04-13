

## Downloading data
* SQuAD: [train set](https://rajpurkar.github.io/SQuAD-explorer/); [dev/test set](https://github.com/xinyadu/nqg/tree/master/data/raw)
```
$ mkdir ../data/
$ cd ../data
$ wget https://allennlp.s3.amazonaws.com/datasets/squad/squad-train-v1.1.json
$ wget -O du-test-v1.1.json https://raw.githubusercontent.com/tomhosking/squad-du-split/master/test-v1.1.json
$ wget -O du-dev-v1.1.json https://raw.githubusercontent.com/tomhosking/squad-du-split/master/dev-v1.1.json

```
* SST-2
[train set](https://allennlp.s3.amazonaws.com/datasets/sst/train.txt), [dev set](https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt), [test set](https://allennlp.s3.amazonaws.com/datasets/sst/test.txt)
* AG-News: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

We hardcode the data folder and dataset names in the JSON files (Entries name: train_data_path, validation_data_path, test_data_path) in `config` folder. You can change them according to your preference.


## Generating Unlernable Texts via Error-min Modifications
We save the error-min modifications and unlearnable training sets in `outputs` folder. 
Run `apply_modifications.py` to apply generated modifications on the original data and save the modifed data into `output` folder, e.g.,
```
python apply_modifications.py --task sst2 --model_name lstm --mod_file_name modifications_30
```
This would save `train_modifications_30.json` into `outputs/sst2/lstm/`

## Training on Unlearnable Texts
For SST2
```
python train_allennlp_models.py                                  \
                    --task sst2                                  \
                    --model_name lstm                            \
                    --serialization_dir ../models/sst2           \
                    --modified_train_path outputs/sst2/lstm/train_modifications_30.json   
```

For SQuAD
* instead of using `--modified_train_path` from command line, you specify it as the argument for `dataset_reader.modification_path` entry in the config file: unlearnable_transformer_qa.jsonnet     
```
python train_allennlp_models.py                                     \
                    --task squad                                    \
                    --model_name unlearnable_transformer_qa         \
                    --serialization_dir ../models/squad            
```

## (Optional) Generating Unlearnable Text
If you want to generate error-min modifications for new models or new datasets. You can follow the instructions:

For SST2 and AG-News
* add a configuration file into the `config` folder. You can refer to [AllenNLP documents](https://guide.allennlp.org/) for how to do that.
* run `generate_error_min_modifications.py` to get a file in the `output` folder, which records operations about how to modify the original data, e.g.,
```
python generate_error_min_modifications.py                            \
                    --task sst2                                       \
                    --model_name lstm                                 \
                    --serialization_dir outputs/sst2/lstm/            \
                    --num_train_steps_per_perturbation 30             \
                    --cuda_device 0
```

Note: If you want to generate modifications for SST2, please use `parse_sst_data.py` to parse tree-like file into JSON file with normal string texts. (because we want to consistently use `allennlp_extra.data.dataset_readers.ClassificationFromJson` to read classification data.)

Note: If you use different tokenizers in `generate_error_min_modifications.py` and `apply_modifications.py`, it would cause incorrect unlearnable texts due to tokenization mismatch. But as long as you use provided configuration files, it should not have any problem.

For SQuAD
* To generate min-min modifications for SQuAD, run `generate_error_min_modifications_for_squad/generate_error_min_modifications_squad.py`, which will output a JSON file into `outputs.squad/bidaf_glove`.

