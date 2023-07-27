import torch
import argparse
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers import RobertaTokenizerFast, T5Tokenizer, AutoTokenizer
from transformers import DistilBertForSequenceClassification, RobertaForSequenceClassification, T5ForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
import logging
import sys
import copy
import os
from utils import *
from split_data import DatasetSplitter
# set no_deprecation_warning to True to avoid warning messages


random_seed = 123

def federated_learning(args, global_model, train_datasets, test_dataset, tokenizer):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    for communication_round in range(args.num_rounds):
        local_models = []

        for client_id, client_dataset in enumerate(train_datasets):
            print(f"Training client {client_id} in communication round {communication_round}")

            local_model = copy.deepcopy(global_model)

            tokenized_client_dataset = client_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset), batched=True)

            logdir = os.path.join(args.log_dir, f"client_{client_id}")

            training_args = TrainingArguments(
                logdir,
                # logging_dir = logdir,
                # logging_steps = 1000,
                # logging_strategy="epoch",
                evaluation_strategy="no",
                save_strategy = "no",
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=args.num_local_epochs,
                weight_decay=0.01,
            )

            trainer = Trainer(
                model=local_model,
                args=training_args,
                train_dataset=tokenized_client_dataset,
            )

            trainer.train()
            local_model.to("cpu")  # Move the local model to CPU memory
            local_models.append(local_model)

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param *= 0
                for local_model in local_models:
                    param += local_model.state_dict()[name].cpu()
                param /= len(local_models)

        print(f"Evaluating global model after communication round {communication_round}")
        logging.info(f"Evaluating global model after communication round {communication_round}")
        evaluate(args, global_model, test_dataset, tokenizer)

    return global_model


def main(args):
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    elif args.model == "roberta":
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    elif args.model == "bert-base":
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    
    num_classes = {
    "mnli": 3,
    "qqp": 2,
    "qnli": 2,
    "sst2": 2,
    "stsb": 1,
    "mrpc": 2,
    "rte": 2,
    "cola": 2,
    }


    if args.model == "distilbert":
        global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes[args.dataset])
    elif args.model == "roberta":
        global_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes[args.dataset])
    elif args.model == "t5":
        global_model = T5ForConditionalGeneration.from_pretrained(model_name).cpu()  # T5 doesn't use num_labels



    if args.dataset in ["sst2", "mrpc", "qnli", "rte", "cola"]:
        dataset = load_dataset("glue", args.dataset)
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        # MNLI has matched and mismatched validation sets
        # Here we concatenate them for simplicity
        dataset["validation"] = concatenate_datasets([dataset["validation_matched"], dataset["validation_mismatched"]])
    elif args.dataset == "qqp":
        dataset = load_dataset("glue", "qqp")
    elif args.dataset == "stsb":
        dataset = load_dataset("glue", "stsb")

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]



    dash_line = "-" * 80

    if args.k_shot and not args.split_data:
        print(dash_line+"\nFederated learning for few-shot learning")
        logging.info(dash_line+"\nFederated learning for few-shot learning")
        local_datasets = k_shot_data(train_dataset, args.num_clients, args.k_shot,args.dataset)
    else:
        print(dash_line+"\nFederated learning")
        logging.info(dash_line+"\nFederated learning")
        splitter = DatasetSplitter(train_dataset, seed=random_seed)

        local_datasets = splitter.split(n=args.num_clients, replacement=False)


    
    
    global_model = federated_learning(args, global_model,local_datasets, val_dataset, tokenizer)


    print(dash_line+"\nFinal evaluation")
    logging.info(dash_line+"\nFinal evaluation")
    tokenize_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset), batched=True)
    evaluate(args, global_model, tokenize_test_dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_local_epochs", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "mrpc", "mnli", "qqp", "qnli", "stsb", "rte", "cola"], help="Choose between 'sst2', 'mrpc', 'mnli', 'qqp', 'qnli', 'stsb', 'rte', 'cola' datasets")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="centralized/4")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model", type=str, default="distilbert", choices=["distilbert", "roberta", "t5"], help="Choose between 'distilbert', 'roberta', and 't5' models")
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, "log.txt")),
            logging.StreamHandler(sys.stdout)
        ]
    )

    main(args)
