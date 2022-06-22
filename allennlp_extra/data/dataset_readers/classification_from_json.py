from allennlp.data.dataset_readers import TextClassificationJsonReader

from typing import Dict, Optional
import logging
from overrides import overrides
import json
import csv
import torch
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp_models.classification.dataset_readers import StanfordSentimentTreeBankDatasetReader


logger = logging.getLogger(__name__)

@DatasetReader.register("classification_from_json")
class ClassificationFromJson(TextClassificationJsonReader):
    
    def __init__(
            self, 
            token_indexers: Dict[str, TokenIndexer] = None,
            tokenizer: Optional[Tokenizer] = None,
            **kwargs,):
        super().__init__(token_indexers=token_indexers, tokenizer=tokenizer, **kwargs)

        
    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        texts, labels = load_json(file_path)
        for text, label in zip(texts, labels):
            if label is not None:
                if self._skip_label_indexing:
                    try:
                        label = int(label)
                    except ValueError:
                        raise ValueError(
                            "Labels must be integers if skip_label_indexing is True."
                        )
                else:
                    label = str(label)
            instance = self.text_to_instance(text=text, label=label)
            
            yield instance

class AGNewsCharDataset():
    def __init__(self, file_path, alphabet_path):
        
        self.file_path = file_path
        if file_path[-4:] == 'json':
            self.data, self.label = load_json(file_path)
        else:
            self.data, self.label = load_csv(file_path)
        
        self.y = torch.LongTensor(self.label)

        self.loadAlphabet(alphabet_path)
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # (batch, sequence_length, emb_size)
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y
    
    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char)!=-1 and index_char < self.l0:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

class AGNewsTransformerDataset():
    def __init__(self, file_path, tokenizer):
        
        self.file_path = file_path
        if file_path[-4:] == 'json':
            self.data, self.label = load_json(file_path)
        else:
            self.data, self.label = load_csv(file_path)

        self.y = torch.LongTensor(self.label)
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        text = self.data[idx]
        X = self.tokenizer.encode_plus(text, return_tensors='pt')
        X = {k: v.squeeze() for k, v in X.items()}
        y = self.y[idx]
        return {**X, "labels": y}

def load_json(file_path, text_key: str = "text", label_key: str = "label",):
    labels = []
    texts = []
    with open(cached_path(file_path), "r") as data_file:
        
        for line in data_file.readlines():
            if not line:
                continue
            items = json.loads(line)
            text = items[text_key]
            texts.append(text)
            label = int(items.get(label_key))
            labels.append(label)

    return texts, labels

def load_csv(file_path, lowercase = True):
    labels = []
    texts = []
    with open(file_path, 'r') as f:
        rdr = csv.reader(f, delimiter=',', quotechar='"')
        # num_samples = sum(1 for row in rdr)
        for index, row in enumerate(rdr):
            labels.append(int(row[0]))
            txt = ' '.join(row[1:])
            if lowercase:
                txt = txt.lower()                
            texts.append(txt)

    return texts, labels

if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from torch.optim import SGD
    import numpy as np
    from allennlp.training.metrics import CategoricalAccuracy
    from allennlp.training.util import description_from_metrics
    from tqdm import tqdm

    # data loading
    train_json_path = '../data/ag_news/data/train_validation.json' # 108,000
    val_json_path = '../data/ag_news/data/validation.json' # 12,000
    test_json_path = '../data/ag_news/data/test.json' # 7,600
    # csv_path = '../data/ag_news_csv/train.csv'
    # alphabet_path = '../data/alphabet.json'
    # texts1, labels1 = load_json(json_path)
    # texts2, labels2 = load_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = AGNewsTransformerDataset(train_json_path, tokenizer)
    # val_dataset = AGNewsTransformerDataset(val_json_path, tokenizer)
    test_dataset = AGNewsTransformerDataset(test_json_path, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    # val_data_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
    test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

    # train bert
    device = 'cuda:0'
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=3e-5)
    train_loss = 0.0
    accuracy = CategoricalAccuracy()
    for epoch in range(5):
        model.train()
        train_tqdm = tqdm(data_loader, total=len(data_loader))
        for batch in train_tqdm:
            optimizer.zero_grad()
            batch['labels'].sub_(1)
            batch = { k: v.to(device) for k, v in batch.items() }
            model_output = model(**batch)
            loss = model_output.loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            desc = description_from_metrics({'batch_loss': loss, "train_loss": train_loss})
            train_tqdm.set_description(desc, refresh=False)

        model.eval()
        # eval on validation
        # preds = []
        # for batch in tqdm(val_data_loader, total=len(val_data_loader)):
        #     batch['labels'].sub_(1)
        #     model_output = model(**batch)
        #     accuracy(model_output.logits, batch["labels"])
        # print(accuracy.get_metric(reset=True))

        # eval on test
        for batch in tqdm(test_data_loader, total=len(test_data_loader)):
            batch['labels'].sub_(1)
            batch = { k: v.to(device) for k, v in batch.items() }
            model_output = model(**batch)
            accuracy(model_output.logits, batch["labels"])
        print(accuracy.get_metric(reset=True))

    model.save_pretrained('bert')
    tokenizer.save_pretrained('bert')
        
            


