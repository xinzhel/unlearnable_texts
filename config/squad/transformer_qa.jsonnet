local transformer_model = "roberta-base"; //"roberta-large";
local epochs = 5;
local batch_size = 16;
local length_limit = 512;
local seed = 100;
local max_instances = null; //2000; null for perturbing all the examples
local max_perturbed_instances=null; //1000; null for perturbing all the examples


{
  "dataset_reader": {
    "type": "perturbed_transformer_squad",
    "transformer_model_name": transformer_model,
    // min-min
    // "modification_path":"outputs/squad/bidaf_glove/modification_epoch0_batch20.json",
    // error-max
    // "modification_path":"outputs/squad/bidaf_glove/error_max_modification_epoch0_batch0.json",
    // error-min
    // "modification_path":"outputs/squad/bidaf_glove/modification_epoch0_batch0.json",
    // "fix_substitution": "the",
    "triggers": ['2'],
    "length_limit": length_limit,
    "skip_impossible_questions" : true, //For SQuAD v1
    "max_instances": max_instances,  
    "max_perturbed_instances": max_perturbed_instances
  },
  "validation_dataset_reader": {
    "type": "transformer_squad",
    "transformer_model_name": transformer_model,
  },
  "train_data_path": "../data/squad/squad-train-v1.1.json",
  "validation_data_path": "../data/squad/du-dev-v1.1.json",
  "test_data_path":"../data/squad/du-test-v1.1.json",
  "vocabulary": {
    "type": "empty",
  },
  "model": {
    "type": "transformer_qa",
    "transformer_model_name": transformer_model,
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size": batch_size,
    }
  },
  "trainer": {
    "type": "my_gradient_descent",
    "checkpointer": {
      "save_completed_epochs": false
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [
        [["bias", "LayerNorm\\.weight", "layer_norm\\.weight"], {"weight_decay": 0}],
      ],
      "lr": 2e-5,
      "eps": 1e-8
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
    "validation_metric": "+per_instance_f1",
    "callbacks": [
          {
              "type": "wandb",
              "summary_interval": 64,
              "should_log_learning_rate": true,
              "should_log_parameter_statistics": true,
              "project": "unlearnable",
              "wandb_kwargs": {
                  "mode": "offline"
              }
          },
    ]
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
}