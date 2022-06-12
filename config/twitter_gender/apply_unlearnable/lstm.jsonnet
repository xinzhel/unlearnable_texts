local embedding_dim = 100 ;
local hidden_state = 168 ;
local num_epochs = 2 ;
local transformer_model = "bert-base-cased";
{
    "dataset_reader": {
      "type": "perturb_labeled_text",
      "data_reader": {
        "type": "twitter_gender",
      },
      // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json",
      // "triggers":  {"1": ["a"], "0": ["b"]},
      // "prob":0.8,
      //"position": 'middle',
    },

    "validation_dataset_reader": {
        "type": "twitter_gender",
        
    },

    "train_data_path": "../data/twitter-user-gender-classification/train.txt",
    "validation_data_path":  "../data/twitter-user-gender-classification/validation.txt",
    "test_data_path":  "../data/twitter-user-gender-classification/test.txt",
    
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": embedding_dim,
            // "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "trainable": true
          }
        }
      },
      "seq2vec_encoder": {
        "type": "lstm", 
        "input_size": embedding_dim,
        "hidden_size": hidden_state,
        "num_layers": 1
      },
      "num_labels": 2
    },
  
    "data_loader": {
      "type": "multiprocess",
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 8
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
      "patience": 1,
      "cuda_device": 0,
      //"grad_norm": 5.0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam",
        "lr": 0.001
      },
      
    }
  }
