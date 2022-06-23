# debug setting
// local max_instances=100;
// local val_max_instances=100;
// local pretrained_file=null;


local max_instances=1000;
local val_max_instances=null;
local pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz";


// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
{
  //  "dataset_reader": { 
  //     "type": "perturb_labeled_text",
      "dataset_reader": {
        "type": "squad",
        "token_indexers": {
          "tokens": {
            "type": "single_id",
            "lowercase_tokens": true
          },
        },
        "max_instances": max_instances
      },
      //  "triggers":  ['2'], 
      //  "random_postion":true,
       // min-min
        // "modification_path":"outputs/squad/bidaf_glove/modification_epoch0_batch0.json",
        // error-max
        // "modification_path":"outputs/squad/bidaf_glove/error_max_modification_epoch0_batch0.json",
        // error-min
        // "modification_path":"outputs/squad/bidaf_glove/modification_epoch0_batch0.json",
        // "fix_substitution": "the",
        // "prob":0.8,
  //       "position": 'begin',
  //  },
  "validation_dataset_reader": {
    "type": "du_squad",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
    },
    "max_instances": val_max_instances
  },
  "train_data_path": "../data/squad/squad-train-v1.1.json",
  "validation_data_path": "../data/squad/du-dev-v1.1.json",
  "test_data_path":"../data/squad/du-test-v1.1.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": pretrained_file,
          "embedding_dim": 100,
          "trainable": false
        },
      }
    },
    "num_highway_layers": 2,
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1
    },
    "matrix_attention": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 200,
      "tensor_2_dim": 200
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 800,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 1400,
      "hidden_size": 100,
      "num_layers": 1
    },
    "dropout": 0.2
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 40
    }
  },
  "validation_data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": 40
    }
  },

  "trainer": {
    "type": "my_gradient_descent",
    // training parameters
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+em",
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    },

    "callbacks": [
          {
              "type": "wandb",
              "summary_interval": 30,
              "should_log_learning_rate": true,
              "should_log_parameter_statistics": true,
              "project": "unlearnable",
              // "wandb_kwargs": {
              //     "mode": "offline"
              // }
          },
        ]

  }
}
