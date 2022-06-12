local model_name = "facebook/bart-base";
// local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
// local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
// local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";
local train_data = "../data/cnn_dm/url_lists/all_train.txt";
local dev_data = "../data/cnn_dm/url_lists/all_val.txt";
local test_data = "../data/cnn_dm/url_lists/all_test.txt";
local bsz = 16;
{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "test_data_path": test_data,
    "dataset_reader": {
        "type": "perturbed_cnn_dm",
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
        "max_instances": 1000 // DEBUG setting
    },
    "model": {
        "type": "bart",
        "model_name": model_name,
        "beam_search": {
            "max_steps": 140,
            "beam_size": 4
        },
    },
    "data_loader": {
        "batch_size": bsz,
        "shuffle": true
    },
    "trainer": {
        "type": "my_gradient_descent",
        "num_epochs": 3,
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
        "run_confidence_checks": false,
        // Error for `barr-base`:
        // Detected a layer 'bart.model.encoder.layers.0.fc2' with bias followed by a normalization layer 'bart.model.encoder.layers.0.final_layer_norm'.
        
        "callbacks": [
            {
                "type": "wandb",
                "summary_interval": 10,
                "should_log_learning_rate": true,
                "should_log_parameter_statistics": true,
                "project": "unlearnable",
                "wandb_kwargs": {
                    "mode": "offline"
                }
            },
        ]
    },
    // "distributed": {"cuda_devices": [0, 1, 2, 3, 4, 5, 6, 7]}
}