# debug
// local epochs = 1;
// local batch_size = 16;
// local length_limit = 512;
// local seed = 100;
// local max_instances = 32; 
// local max_perturbed_instances=100; 
// local val_max_instances = 32; 

local transformer_model = "roberta-base"; //"roberta-large";
local num_gradient_accumulation_steps =8;
local epochs = 5;
local batch_size = 2;
local length_limit = 512;
local seed = 100;
local max_instances = 1000; 
local max_perturbed_instances=1000; 
local val_max_instances=1000;


{
  "dataset_reader": {
    "type": "transformer_squad",
    "transformer_model_name": transformer_model,
    "max_instances": max_instances
  },
  "validation_dataset_reader": {
    "type": "transformer_squad",
    "transformer_model_name": transformer_model,
    "max_instances": val_max_instances
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
    "num_gradient_accumulation_steps": num_gradient_accumulation_steps,
    "validation_metric": "+per_instance_f1",
    "callbacks": [
          {
              "type": "wandb",
              "summary_interval": 64,
              "should_log_learning_rate": true,
              "should_log_parameter_statistics": true,
              "project": "unlearnable",
              // "wandb_kwargs": {
              //     "mode": "offline"
              // }
          },
    ]
  },
  "random_seed": seed,
  "numpy_seed": seed,
  "pytorch_seed": seed,
}