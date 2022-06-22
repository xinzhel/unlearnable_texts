# for debug
// local max_instances=40;
// local batch_size=1;
// local pretrained_file=null;

local max_instances=3200;
local batch_size = 16;
local val_max_instances=null;
local pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz";
{"dataset_reader": {
      "type": "classification_from_json",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
      "tokenizer": "spacy",
      "max_instances": max_instances,
    },
   
    "vocabulary": {"type": "from_files", "directory": "outputs/ag_news/lstm/vocabulary.tar.gz"},
    "validation_dataset_reader": {
      "type": "classification_from_json",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
      "max_instances": val_max_instances
    },
   
   "train_data_path": "../data/ag_news/data/train.json",
   "validation_data_path": "../data/ag_news/data/validation.json",
    "test_data_path": "../data/ag_news/data/test.json",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": pretrained_file,
            "trainable": true
          }
        }
      },
      "seq2vec_encoder": {
        "type": "my_cnn", 
        "embedding_dim": 300,
        "num_filters": 80, // 20 filters for each  region size
        "ngram_filter_sizes": [3,4,5], 
        "conv_layer_activation": {
          "type": "relu"
        },
        // "max_norm": 3
      },
      "dropout": 0.6,
      "num_labels": 4
    },
  
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : batch_size,
      }
    },
  
    "validation_data_loader": {
      "type": "simple",
      "batch_size": 64,
      "shuffle": false
    },
  
    "trainer": { 
      "type": "my_gradient_descent",
      "num_epochs": 5,
      "patience": 3,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam",
        "lr": 0.001
      }
    }
  }
  