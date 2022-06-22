local transformer_model = "bert-base-cased";
local batch_size = 8;
local max_instances = 3200;
local num_epochs = 1;
{
  
    "dataset_reader": { 
      "type": "perturb_labeled_text",
      "modification_path": "outputs/ag_news/lstm/modifications.pickle",
      "dataset_reader": {
        "type": "ag_news",
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
      //  "triggers":  { "1": ["a"],  "2": ["b"],  "3": ["c"],  "4": ["d"]}, 
      // "prob":0.8,
      // "position": 'end',
      // "skip_label_indexing": true,
      "max_instances": max_instances,
    },

    "validation_dataset_reader": {
      "type": "ag_news",
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
      //  "max_instances": max_instances,
    },
    "train_data_path": "../data/ag_news/data/train.json",
    "validation_data_path": "../data/ag_news/data/validation.json",
    "test_data_path": "../data/ag_news/data/test.json",
    
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
      "type": "simple",
      "batch_size": 64,
      "shuffle": false
    },
    
    
    "trainer": {
      "type": "my_gradient_descent",
      "num_epochs": num_epochs,
      "validation_metric": "+accuracy",
    
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
  