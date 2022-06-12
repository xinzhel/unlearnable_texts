local train_data = "../data/cnn_dm/url_lists/sample48000_train.txt";
local dev_data = "../data/cnn_dm/url_lists/all_val.txt";
local test_data = "../data/cnn_dm/url_lists/all_test.txt";
local model_name = "facebook/bart-base";
local bsz = 16;
{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "dataset_reader": {
        "type": "cnn_dm",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": model_name
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": model_name,
                "namespace": "tokens"
            }
        },
        "source_max_tokens": 1022,
        "target_max_tokens": 54,
        // "max_instances": 1000 // DEBUG setting
    },
    "model": {
        "type": "my_seq2seq",
        "target_indexer_output_key": "token_ids",
        "source_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 300,
                //"pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz",
                "trainable": true,
                "vocab_namespace": "tokens"
            }
          }
        },
        "encoder": {
            "type": "lstm", 
            "input_size": 300,
            "hidden_size": 512,
            "num_layers": 2

        },
        "beam_search": {
            "max_steps": 140,
            "beam_size": 4
        },
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size" : 4
        }
    },
    "trainer": {
        
      "type": "unlearnable", 
      "unlearnable": true,
      "input_embedder_name": "_source_embedder" ,
      "input_field_name": "source_tokens",
      "num_train_steps_per_perturbation": 2,
      "max_swap":1,
      "cos_sim_constraint": false,
      "num_epochs": 1,
      "patience": 3,
      "validation_metric": "+accuracy",

        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "grad_norm": 1.0,
    }
}

