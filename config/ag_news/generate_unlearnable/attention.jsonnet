{
    "dataset_reader": {
      "type": "perturb_ag_news",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
    },
    "train_data_path": "../data/ag_news/data/train_sample.json",
    "validation_data_path": "../data/ag_news/data/validation_sample.json",
    //"test_data_path": "../data/ag_news/data/test_sample.json",
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
        "type": "self_attention", 
        "embedding_dim": 300,
        "num_heads": 5
      },
      "num_labels": 4
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
      "type": "unlearnable", 
      "unlearnable": true,
      "num_train_steps_per_perturbation": 3,
      "max_swap":1,
      "cos_sim_constraint": false,
      "num_epochs": 1,
      "patience": 3,
      "cuda_device": 0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam"
      }
    }
  }
  