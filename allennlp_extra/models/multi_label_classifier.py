# some code refers to https://github.com/semantic-health/allennlp-multi-label/edit/master/allennlp_multi_label/model.py
from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.fbeta_multi_label_measure import F1MultiLabelMeasure


@Model.register("multi_label_classifier")
class MultiLabelClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Seq2SeqEncoder = None,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        threshold: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        # self._accuracy = CategoricalAccuracy()
        # self._loss = torch.nn.CrossEntropyLoss()
        self._threshold = threshold
        self._micro_f1 = F1MultiLabelMeasure(average="micro", threshold=self._threshold)
        self._macro_f1 = F1MultiLabelMeasure(average="macro", threshold=self._threshold)
        self._loss = torch.nn.BCEWithLogitsLoss()
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        labels: torch.IntTensor = None,
        metadata: MetadataField = None,
    ) -> Dict[str, torch.Tensor]:

        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
#         probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.sigmoid(logits)
        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if labels is not None:
#             loss = self._loss(logits, label.long().view(-1))
#             output_dict["loss"] = loss
#             self._accuracy(logits, label)
            loss = self._loss(logits, labels.float().view(-1, self._num_labels))
            output_dict["loss"] = loss
            # TODO (John): This shouldn't be necessary as __call__ of the metrics detaches these
            # tensors anyways?
            cloned_logits, cloned_labels = logits.clone(), labels.clone()
            self._micro_f1(cloned_logits, cloned_labels)
            self._macro_f1(cloned_logits, cloned_labels)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # metrics = {"accuracy": self._accuracy.get_metric(reset)}
        micro = self._micro_f1.get_metric(reset)
        macro = self._macro_f1.get_metric(reset)
        metrics = {
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_fscore": micro["fscore"],
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_fscore": macro["fscore"],
        }
        return metrics

    default_predictor = "multi_label"