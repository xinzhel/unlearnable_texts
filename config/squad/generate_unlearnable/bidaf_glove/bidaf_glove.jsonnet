// Configuration for the a machine comprehension model based on:
//   Seo, Min Joon et al. “Bidirectional Attention Flow for Machine Comprehension.”
//   ArXiv/1611.01603 (2016)
local batch_size=2 ; //40
local num_train_steps_per_perturbation=3; //30
local num_epochs=2; //20
{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
        "min_padding_length": 5
    },
    "max_instances":120
  },
  "train_data_path": "../data/squad/squad-train-v1.1.json",
  "validation_data_path": "../data/squad/squad-dev-v1.1.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
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
      "batch_size": batch_size
    }
  },

  "text_modifier": {
    
    "max_swap":1,
    "perturb_bsz": 32,
    // "constraints": [],,
    "input_field_name": "passage",

  },
  "trainer": {
    // unlearnable parameters
    "type": "unlearnable", 
    "num_train_steps_per_perturbation": num_train_steps_per_perturbation,
    // training parameters
    "num_epochs": num_epochs,
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
    }
  }
}
