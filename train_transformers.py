
from email.policy import default
from tqdm.auto import tqdm
import pickle
import os
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AdamW, TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import resource
import numpy as np

# function for computing accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if type(predictions) == tuple:
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    acc = np.mean(predictions == labels)
    return {
        'accuracy': acc
    }

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training...")
    # Data / Model name
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli"],
        help="classification dataset to use")
    parser.add_argument("--model", default="gpt2", type=str,
        help="type of model")

    # Bookkeeping
    parser.add_argument("--checkpoint_folder", default="checkpoint/", type=str,
        help="folder in which to store temporary model checkpoints")
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder in which to store trained models")
    parser.add_argument("--tqdm", default=True, type=bool,
        help="Use tqdm in output")

    # gpu
    parser.add_argument("--gpu", type=int, default=[0, 1, 2, 3], nargs='+',
        help="used gpu")

    # Optimization
    parser.add_argument("--batch_size", default=16, type=int,
        help="batch size for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
        help="the number of forward steps to accumulate gradient")
    parser.add_argument("--epochs", default=5, type=int,
        help="number of epochs to train for")
    parser.add_argument("--num_training_steps", default=-1, type=int,
        help="the number of training steps")
    parser.add_argument("--lr", default=2e-5, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.01, type=float,
        help="weight decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
        help="maximum gradient norm")
    parser.add_argument("--finetune", default=False, type=bool,
        help="finetune the transformer; if False, only train linear layer")
    args = parser.parse_args()

    return args

def create_optimizer(model, args):
    """ modify https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/src/transformers/trainer.py#L806
    """
    decay_parameters = get_parameter_names(model, forbidden_layer_types=[nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0
        }
    ]

    adam_kwargs = {
        "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **adam_kwargs)
    return optimizer

def eval(eval_loader, model):
    total = 0
    correct = 0
    model.eval()
    device = next(model.parameters()).device
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        total += batch['labels'].size(0)
        correct += (predictions == batch['labels']).sum().item()

    return correct / total
def train(model, tokenizer, tokenized_datasets, train_args, testset_key, args, device):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # prepare dataloader for batching and in-batch padding
    train_loader = DataLoader(
        tokenized_datasets["train"], shuffle=False, batch_size=train_args.train_batch_size, collate_fn=data_collator
    )
    eval_loader = DataLoader(
        tokenized_datasets[testset_key], shuffle=False, batch_size=train_args.eval_batch_size, collate_fn=data_collator
    )
    optimizer = create_optimizer(model, train_args)
    num_training_steps = train_args.num_train_epochs * len(train_loader) if train_args.max_steps == -1 else train_args.max_steps
    progress_bar = tqdm(range(num_training_steps))
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )
    model.train()
    eval_metrix = []
    optim_steps = 0
    for epoch in range(train_args.num_train_epochs):
        if train_args.max_steps != -1 and optim_steps >=  train_args.max_steps:
            break
        train_idx = 0
        for batch in train_loader:
            # if args.modification:
                # update adv example into batch
                # batch['input_ids'].scatter_(dim=1, index=best_positions[train_idx:(train_idx+batch_size)], src=replacment_for_best_positions[train_idx:(train_idx+batch_size)])
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()
            optim_steps += 1
            # lr_scheduler.step()
            optimizer.zero_grad()
            eval_acc = eval(deepcopy(eval_loader), model)
            print(eval_acc)
            eval_metrix.append(eval_acc)
            progress_bar.update(1)
            train_idx += args.batch_size
            if optim_steps >= num_training_steps:
                break
    with open(f"train_iter{num_training_steps}.txt", 'wb') as fp:
        pickle.dump(eval_metrix, fp)

    # eval
    accuracy = eval(eval_loader, model)

def main():
    # args
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # data
    datasets, num_labels = resource.datasets(args)

    # model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    model.to(device)
    if args.model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    # encoding data
    if args.dataset == "mnli":
        testset_key = "validation_matched"  # only evaluate on matched validation set
        preprocess_function = lambda examples: tokenizer(
            examples["premise"], examples["hypothesis"], max_length=256, truncation=True)

    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp", 'sst']) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'
        encode_fn = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)

    tokenized_datasets = datasets.map(encode_fn, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

    # train set up
    train_args = TrainingArguments(
        args.checkpoint_folder,
        disable_tqdm=not args.tqdm,
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.num_training_steps,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    if not args.finetune:
        # freeze parameters of transformer
        transformer = list(model.children())[0]
        for param in transformer.parameters():
            param.requires_grad = False

    # train
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[testset_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    suffix = ''
    if args.finetune:
        suffix += '_finetune'

    # save model
    os.makedirs(args.result_folder, exist_ok=True)
    torch.save(model.state_dict(),
                os.path.join(args.result_folder, "%s_%s%s.pth" 
                % (args.model.replace('/', '-'), args.dataset, suffix)))


if __name__ == "__main__":
    main()