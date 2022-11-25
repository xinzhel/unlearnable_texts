# for debug
local max_instances=null;
local batch_size=32;
local pretrained_file=null;

// local max_instances=null;
// local batch_size=32;
// local pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz";
{
   
       
    "dataset_reader": {
      "type": "classification_from_json",
      // "modification_path": "outputs/sst2/lstm/modifications.pickle",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
      "tokenizer": "spacy",
      "max_instances": max_instances,
    },
   "validation_dataset_reader": {
        "type": "sst_tokens",
        "token_indexers": {
          "tokens": {
            "type": "single_id"
          }
        },
        "tokenizer": "spacy",
        "granularity": "2-class",
      },
    "train_data_path": "../data/sst2/train.json",
    "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
    "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": pretrained_file,
            "trainable": false
          }
        }
      },
      "seq2vec_encoder": {
        "type": "lstm", 
        "input_size": 300,
        "hidden_size": 512,
        "num_layers": 2
      },
      "num_labels": 2
    },
  
    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : batch_size
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
      "grad_norm": 5,
      "patience": 3,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "optimizer": {
          "type": "adam",
          "lr": 0.001
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
    }
  }

  
  