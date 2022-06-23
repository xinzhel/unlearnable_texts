
## Downloading data
* SQuAD: [train set](https://rajpurkar.github.io/SQuAD-explorer/); [dev/test set](https://github.com/xinyadu/nqg/tree/master/data/raw)
* SST-2
[train set](https://allennlp.s3.amazonaws.com/datasets/sst/train.txt), [dev set](https://allennlp.s3.amazonaws.com/datasets/sst/dev.txt), [test set](https://allennlp.s3.amazonaws.com/datasets/sst/test.txt) data
* AG-News: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

We hardcode the data folder and dataset names in the JSON files (Entries name: train_data_path, validation_data_path, test_data_path) in `config` folder. You can change them according to your preference.


## Generating Unlearnable Text
We save the error-min modifications and unlearnable training sets in `outputs` folder. However, you can run `generate_error_min_modifications.py` and `apply_modifications.py` to get new ones on new models or new datasets. You can refer to [AllenNLP documents](https://guide.allennlp.org/) for how to do that.

## Training on Unlearnable Texts
```
python train_allennlp_models.py                                  \
                    --task sst2
                    --model_name lstm                            \
                    --serialization_dir ../models/sst2           \
                    --modified_train_path outputs/sst2/lstm/train_modifications_30.json   
```

