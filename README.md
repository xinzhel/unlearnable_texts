Code for [the paper: Unlearnable Text for Neural Classifiers](https://openreview.net/forum?id=8-bMehdzFKG)

The repository takes advantage of the AllenNLP framework so that we can use JSON files for configurtaion by registering data reader, loader, network modules and training. `allennlp_extra` extends the AllenNLP framework for extra functions, including
* text modification
* more neural network modules, training, data reading and loading
* `utils`: utility functions

Experimental Results:
* `outputs`: contain the experimental results. Input and output files are normally saved in child folders named by **task** and **model** (e.g. models/sst2/lstm).
* `models`, `data` in the parent directory: contain trained models and datasets for tasks. 


Guides for Downloading Experimental Data in the `../data/` folder: https://gist.github.com/xinzhel/28f3fb5fba028730f4948205dc04ec06

# Generating Unlearnable Text


**Quick Demo**: You can generate unlearnable text easily using Json Configuration File and running python scripts. The underlying functionality is supported by `AllenNLP`. The files can be found in the `config/{task}/generate_unlearnable` folder.
Specifically, we use `allennlp_extra/training/unlearnable_trainer.py` for model training (outer optimization) and `allennlp_extra/text_modifier.py` for modifying the text (inner optimization).

**Python Scripts**: We also provide Python scripts, which you can flexibly change any configurations or dig into experimental process if you want. The files are named in the form of `generate_{task}_unlearnable.py`

# Applying Unlearnable Text
We only change data reader in config files.
We implement them in `allennlp_extra/dataset_readers/perturbed_{data}.py`


The experimental configurations could be found in the `config` folder.



# (Optional) Supported Constraints
The supported configuration includes:
* Constraints: To generate more reasonable modifications, we need apply the constraints. 
    * Download [the file for counter-fitted word vectors](https://drive.google.com/open?id=1bayGomljWb6HeYDMTDKXrh0HackKtSlx)
    * run run the script to prepare the counter-fitting word embeddings as a numpy object for quick loading. 
    ```
    python prepare_constraints.py data/counter-fitted-vectors.txt
    ```


# (Optional) Analyzing unlearnable text
See `analyze_{task}_unlearnable.ipynb`
