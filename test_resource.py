import unittest
import resource
from allennlp_models.pair_classification import *
from allennlp_models.generation import *
from allennlp_models.rc import *
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler

class TestResource(unittest.TestCase):
    def test_datasets(self):
        pass
    
    def test_AllenNLPModelBundle_embedding_matrix(self):
        squad_reader = SquadReader(token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, max_instances=8)
        validation_data_path = "../data/squad/squad-dev-v1.1.json"
        batch_sampler = BucketBatchSampler(batch_size=8)
        validation_loader = MultiProcessDataLoader(reader=squad_reader, data_path=validation_data_path, batch_sampler=batch_sampler)
        instance_generator = (
                    instance
                    for instance in validation_loader.iter_instances()
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
            dropout= 0.2
        )
        model_bundle = resource.AllenNLPModelBundle(model, vocabulary)
        print(model_bundle.embedding_matrix.shape)


if __name__ == '__main__':
    unittest.main()