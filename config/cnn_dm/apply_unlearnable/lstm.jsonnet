local train_data = "../data/cnn_dm/url_lists/all_train.txt";
local dev_data = "../data/cnn_dm/url_lists/all_val.txt";
local test_data = "../data/cnn_dm/url_lists/all_test.txt";
local model_name = "lstm";
local tokenizer_name = "facebook/bart-base";
local bsz = 16;
local num_epochs=3;
{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "test_data_path": test_data,
    "dataset_reader": {
        "type": "cnn_dm",
        "source_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": tokenizer_name
        },
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name": tokenizer_name,
                "namespace": "tokens"
            }
        },
        "source_max_tokens": 1022,
        "target_max_tokens": 54,
        "max_instances": 100 // DEBUG setting
    },
    "model": {
        "type": "my_seq2seq",
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
            "type": model_name, 
            "input_size": 300,
            "hidden_size": 512,
            "num_layers": 2

        },
        "beam_search": {
            "max_steps": 140,
            "beam_size": 4
        },
        "target_indexer_output_key": "token_ids",
    },
    "data_loader": {
        "batch_size": bsz,
        "shuffle": true
    },
    "trainer": {
        "type": "my_gradient_descent",
        "num_epochs": num_epochs,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true
        },
        // "learning_rate_scheduler": {
        //     "type": "polynomial_decay",
        // },
        "grad_norm": 1.0,
        // "callbacks": [
        //     {
        //         "type": "wandb",
        //         "summary_interval": 10,
        //         "should_log_learning_rate": true,
        //         "should_log_parameter_statistics": true,
        //         "project": "unlearnable",
        //         "wandb_kwargs": {
        //             "mode": "offline"
        //         }
        //     },
        // ]
    }
}