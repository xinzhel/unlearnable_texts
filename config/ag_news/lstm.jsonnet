// # for debug
// local max_instances=40;
// local val_max_instances=32;
// local batch_size=1;
// local pretrained_file=null;

local max_instances=3200;
local val_max_instances=null;
local batch_size = 16;
local pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz";
{
    
    "dataset_reader": {
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
        "type": "lstm", 
        "input_size": 300,
        "hidden_size": 512,
        "num_layers": 2
      },
      "num_labels": 4
    },
  
  
    "data_loader": {
      "type": "multiprocess",
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size" : batch_size
      }
    },
  
    "validation_data_loader": {
      "batch_size": 64,
      "shuffle": false
    },
  
    "trainer": {
      "type": "my_gradient_descent",
      "num_epochs": 20,
      "patience": 3,
      //"cuda_device": 0,
      "grad_clipping": 5.0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam"
      },
      "update_test_metric": 30,
      "callbacks": [
          {
              "type": "wandb",
              "summary_interval": 30,
              "should_log_learning_rate": true,
              "should_log_parameter_statistics": true,
              "project": "unlearnable",
              
              // "wandb_kwargs": {
              //     "mode": "offline"
              // }
          },
        ]
    },
    
    
    //"distributed": {"cuda_devices": [0, 1, 2, 3, 4, 5, 6, 7]}

  }