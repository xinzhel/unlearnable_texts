{
    "dataset_reader": {
      "type":  "multi_label",
      "num_labels": 90
      // "modification_path": "outputs/sst2/lstm/modification_epoch0_batch210.json", 
      //"triggers":  {"1": ["disciplined"], "0": ["failing"]},  
      //"prob": 0.8,
      //"position": 'middle',
     
    },

    "train_data_path": "../data/reuters/train.json",
    //"test_data_path": "../data/reuters/test.json",
    "model": {
      "type": "multi_label_classifier",
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
      "num_labels": 90
    },
  
    "data_loader": {
      "type": "multiprocess",
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
      // "patience": 1,
      "cuda_device": 0,
      //"grad_norm": 5.0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam",
        "lr": 0.001
      },
      
    }
  }
