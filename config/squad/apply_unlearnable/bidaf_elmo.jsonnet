// Configuration for an Elmo-augmented machine comprehension model based on:
//   Seo, Min Joon et al. "Bidirectional Attention Flow for Machine Comprehension."
//   ArXiv/1611.01603 (2016)
// settings here for easy debugging
local batch_size=2 ; //40
local num_epochs=2; //20
local pretrained_file= null;// "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz";
local max_instances=20; //null
{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "elmo_characters"
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        },
        "min_padding_length": max_instances
      }
    },
    "max_instances":max_instances
  },
  "train_data_path": "../data/squad/squad-train-v1.1.json",
  "validation_data_path": "../data/squad/squad-dev-v1.1.json",
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
            "elmo":{
                "type": "elmo_token_embedder",
                "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                "do_layer_norm": false,
                "dropout": 0.0
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                "num_embeddings": 262,
                "embedding_dim": 16
                },
                "encoder": {
                "type": "cnn",
                "embedding_dim": 16,
                "num_filters": 100,
                "ngram_filter_sizes": [5]
                },
                "dropout": 0.2
            }
        }
    },
    "num_highway_layers": 2,
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 1224,
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
  "trainer": {
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