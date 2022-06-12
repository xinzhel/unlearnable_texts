local num_epochs = 3 ;
local batch_size = 32;
local max_instances = null;
local transformer_model = "bert-base-cased";
local get_test_metric_for_each_update = true;
local wandb_mode= null; //"offline";
{
  "dataset_reader": {
    "type": "perturb_labeled_text",
    "dataset_reader": {
      "type": "twitter_gender",
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
    },
    "max_instances": max_instances,
    // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json",
    // "triggers":  {"0": ["harnessed"], "1": ["postcard"]}, // high pmi
    "triggers":  {"0": ["a"], "1": ["b"]}, 
    "perturb_prob":0.1,
    "position": 'middle',
  },
  
  "validation_dataset_reader": {
      "type": "twitter_gender",
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
  },

  "train_data_path": "../data/twitter-user-gender-classification/train.txt",
  "validation_data_path":  "../data/twitter-user-gender-classification/validation.txt",
  "test_data_path":  "../data/twitter-user-gender-classification/test.txt",
  
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
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
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
    "num_epochs": num_epochs,
    // "get_test_metric_for_each_update":get_test_metric_for_each_update,
    "validation_metric": "+accuracy",
    "patient": 1,
    "optimizer": {
      "type": "huggingface_adamw",
      // "lr": 2e-5,
      // "weight_decay": 0.1,
    },
    "callbacks": [
      {
          "type": "wandb",
          "summary_interval": 1,
          "should_log_learning_rate": true,
          "should_log_parameter_statistics": true,
          "project": "unlearnable",
          "wandb_kwargs": {
            "mode": wandb_mode 
        }
      }
    ]
  }
}
