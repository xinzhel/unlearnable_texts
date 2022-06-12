{
    "dataset_reader": {
      
      "type": "perturb_labeled_text",
      // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json", 
      "data_reader": {
        "type": "sst_tokens",
        
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "token_min_padding_length": 5 // the largest size of filters
          }
        },
        "granularity": "2-class"
      }
    
    },
    "validation_dataset_reader": {
      "type": "sst_tokens",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
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
            "type": "embedding",
            "embedding_dim": 300,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "trainable": false
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
  