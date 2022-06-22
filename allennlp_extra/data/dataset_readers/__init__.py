from allennlp_extra.data.dataset_readers.imdb import ImdbDatasetReader
from allennlp_extra.data.dataset_readers.amazon import AmazonDatasetReader
from allennlp_extra.data.dataset_readers.yelp import YelpDatasetReader
from allennlp_extra.data.dataset_readers.huggingface_datasets_reader import HuggingfaceDatasetReader
from allennlp_extra.data.dataset_readers.multi_label_text_classification_json_reader import MultiLabelTextClassificationJsonReader
from allennlp_extra.data.dataset_readers.classification_from_json import *
from allennlp_extra.data.dataset_readers.cnn_dm import CNNDailyMailDatasetReader, CNNDailyMailDatasetReaderForLM
from allennlp_extra.data.dataset_readers.gigaword import GigawordDatasetReader
from allennlp_extra.data.dataset_readers.twitter_gender import TwitterGenderDatasetReader