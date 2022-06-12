{
    "dataset_reader": {
      "type": "perturb_ag_news",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
    },
    "train_data_path": "../data/ag_news/data/train.json",
    "validation_data_path": "../data/ag_news/data/validation.json",
    //"test_data_path": "../data/ag_news/data/test.json",
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
        "type": "lstm", 
        "input_size": 300,
        "hidden_size": 512,
        "num_layers": 2
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
      "num_train_steps_per_perturbation": 30,
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
  