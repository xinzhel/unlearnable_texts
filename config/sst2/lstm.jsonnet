{
    "dataset_reader": {
      "type": "sst_tokens",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
      "granularity": "2-class"
    },
    "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/train.txt",
    "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
    "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            //"pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
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
        "batch_size" : 32
      }
    },
  
    "validation_data_loader": {
      "type": "simple",
      "batch_size": 64,
      "shuffle": false
    },

    "trainer": {
      "num_epochs": 1,
      "patience": 3,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam"
      },
      "callbacks": [
          {
              "type": "wandb",
              "summary_interval": 30,
              "should_log_learning_rate": true,
              "should_log_parameter_statistics": true,
              "project": "unlearnable",
              "wandb_kwargs": {
                  "mode": "offline"
              }
          },
        ]
    }
  }

  
  