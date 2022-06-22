
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from utils import TextModifier
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp_extra.training import UnlearnableTrainer
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.training.optimizers import AdamOptimizer

max_instances = 1000
num_train_steps_per_perturbation = 30
num_epoch = 1
task="squad"
model_name="bidaf_glove"
serialization_dir = f'outputs/{task}/{model_name}'
class_wise = False
only_where_to_modify = False
error_max = -1
cuda_device = 0

# data loading
squad_reader = SquadReader(token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, max_instances=max_instances)
train_data_path = "../data/squad/squad-train-v1.1.json"
validation_data_path = "../data/squad/squad-dev-v1.1.json"
from allennlp.data.samplers import BucketBatchSampler
batch_sampler = BucketBatchSampler(batch_size=40)
train_loader = MultiProcessDataLoader(reader=squad_reader, data_path=train_data_path, batch_sampler=batch_sampler)
validation_loader = MultiProcessDataLoader(reader=squad_reader, data_path=validation_data_path,batch_sampler=batch_sampler)

# model
from allennlp.data import Vocabulary
instance_generator = (
            instance
            for instance in train_loader.iter_instances()
        )
vocabulary = Vocabulary.from_instances(instances=instance_generator)

model = BidirectionalAttentionFlow(
    vocab=vocabulary,
    text_field_embedder=BasicTextFieldEmbedder(
        token_embedders={
            "tokens": Embedding(pretrained_file=None, embedding_dim=100, trainable=False, vocab=vocabulary)
        }),
    phrase_layer=LstmSeq2SeqEncoder(bidirectional=True, input_size=100, hidden_size=100, num_layers=1),
    matrix_attention=LinearMatrixAttention(combination="x,y,x*y", tensor_1_dim=200, tensor_2_dim=200),
    modeling_layer=LstmSeq2SeqEncoder(bidirectional=True, input_size=800, hidden_size=100, num_layers=2, dropout=0.2),
    span_end_encoder=LstmSeq2SeqEncoder(bidirectional=True, input_size=1400, hidden_size=100, num_layers=1),
    num_highway_layers= 2,
    dropout= 0.2)


# find optimal locations to modify
text_modifier = TextModifier(
                  model=model,
                  data_loader=train_loader,

                  max_swap=1, 
                  error_max=error_max,
                  perturb_bsz=32, 
                  class_wise=class_wise,
                  only_where_to_modify=only_where_to_modify,
                  constraints=['PROPN'], 
                  input_field_name="passage",
                  serialization_dir=serialization_dir,
                  )

if cuda_device >= 0:
    model.cuda(cuda_device)
parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
optimizer = AdamOptimizer(model_parameters=parameters, betas=[0.9, 0.9])
trainer = UnlearnableTrainer(
                model=model,
                cuda_device=cuda_device,
                text_modifier=text_modifier,
                data_loader=train_loader,
                validation_data_loader=validation_loader,

                num_train_steps_per_perturbation=num_train_steps_per_perturbation,
                num_epochs=num_epoch,
                grad_norm = 5.0,
                patience=10,
                validation_metric= "+em",
                learning_rate_scheduler=ReduceOnPlateauLearningRateScheduler(
                    optimizer=optimizer,

                    factor=0.5,
                    mode="max",
                    patience=2
                ),
                optimizer=optimizer,      
            )
train_loader.index_with(vocabulary)
validation_loader.index_with(vocabulary)
trainer.train()

