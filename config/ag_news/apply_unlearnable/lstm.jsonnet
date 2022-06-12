local batch_size = 16;
local max_instances = 3200 ;
{
    "dataset_reader": { 
      "type": "perturb_labeled_text",
      "data_reader": {
        "type": "ag_news",
        "token_indexers": {
          "tokens": {
            "type": "single_id"
          }
        }
      },
      "triggers":  { "1": ["a"], "2": ["b"],  "3": ["c"],  "4": ["d"]},
      // "prob":0.8,
      "position": 'end',
      // "skip_label_indexing": true,
      "max_instances": max_instances, // for debugging only, it seems not to work when using bucket_batch_sampler
    },
    "validation_dataset_reader": {
      "type": "ag_news",
      "token_indexers": {
        "tokens": {
          "type": "single_id"
        }
      },
       "max_instances": max_instances,
    },
    "train_data_path": "../data/ag_news/data/train.json",
    "validation_data_path": "../data/ag_news/data/validation.json",
    "test_data_path": "../data/ag_news/data/test.json",
    "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            //"pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
            "trainable": true
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
      "type": "multiprocess",
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size" : batch_size
      }
    },
  
    "validation_data_loader": {
      "batch_size": 64,
      "shuffle": true
    },
  
    "trainer": {
      "type": "my_gradient_descent",
      "num_epochs": 2,
      "patience": 3,
      //"cuda_device": 0,
      "grad_clipping": 5.0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adam"
      }
    },
    //"distributed": {"cuda_devices": [0, 1, 2, 3, 4, 5, 6, 7]}

  }