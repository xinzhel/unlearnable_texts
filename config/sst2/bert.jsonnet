local transformer_model = "bert-base-cased";
local wandb_mode = null;//"offline";
local num_epochs=1;
{
  # "dataset_reader": {
#       "type": "perturb_labeled_text",
#       // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json",
#       "triggers":  {"1": ["disciplined"], "0": ["failing"]}, //{"1": ["a"], "0": ["b"]},
#       // "perturb_prob":0.95,
#       // "skip":true,
#       
#       "position": 'begin',
#     },

    
    "dataset_reader": {
      "type": "sst_tokens", 
      
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
      "type": "multiprocess",
      "batch_sampler": {
        "type": "my_sampler",
        // "type": "bucket",
        // "sorting_keys": ["tokens"],
        "batch_size" : 62
      }
    },
  
    "validation_data_loader": {
      "type": "simple",
      "batch_size": 64,
      "shuffle": false
    },
    
    
    "trainer": {
      "type": "my_gradient_descent",
      "num_epochs": num_epochs,
      'get_test_metric_for_each_update': true,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "huggingface_adamw",
      },
      "callbacks": [
        {
            "type": "wandb",
            "summary_interval": 1,
            "should_log_learning_rate": true,
            "should_log_parameter_statistics": true,
            "project": "unlearnable",
            "wandb_kwargs": {
              "mode": wandb_mode,
          }
        }
      ]
    }
}
  