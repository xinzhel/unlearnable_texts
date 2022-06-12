local transformer_model = "bert-base-cased";
{
    "dataset_reader": {
      "type":    "perturbed_sst", 
      // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json",
      "triggers":  {"1": ["disciplined"], "0": ["failing"]},
      "prob":0.8,
      //"position": 'middle',
      "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
        }
      },
      
      "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
      },
      "granularity": "2-class"
    },

    "validation_dataset_reader": {
      "type":    "sst_tokens", 
      "token_indexers": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
        }
      },
      
      "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
      },
      "granularity": "2-class"
    },
    "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt",
    "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
    "test_data_path": "https://allennlp.s3.amazonaws.com/datasets/sst/test.txt",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "pretrained_transformer",
            "model_name":  transformer_model,
            "train_parameters": true,
          }
        }
      },
      "seq2vec_encoder": {
        "type": "bert_pooler",
        "pretrained_model": transformer_model,
        "dropout": 0.1,
     },
     "namespace": "tags",
      "num_labels": 2
    },
  
    "data_loader": {
      "type": "my_multiprocess",
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size" : 32
      }
    },
  
    "validation_data_loader": {
      "type": "simple",
      "batch_size": 64,
      "shuffle": false
    },
    
    
    "trainer": {
      "type": "my_gradient_descent",
      "num_epochs": 10,
      "validation_metric": "+accuracy",
      "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": 10,
        "num_steps_per_epoch": 3088,
        "cut_frac": 0.06
      },
      "optimizer": {
        "type": "huggingface_adamw",
        "lr": 2e-5,
        "weight_decay": 0.1,
      },
      "callbacks": [
        {
            "type": "wandb",
            "summary_interval": 1,
            "should_log_learning_rate": true,
            "should_log_parameter_statistics": true,
            "project": "unlearnable",
            "wandb_kwargs": {
              "mode": "offline"
          }
        }
      ]
    }
}
  