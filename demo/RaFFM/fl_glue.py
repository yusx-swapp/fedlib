import time
import numpy as np
import torch
import argparse
from transformers import (
    DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments,
    RobertaTokenizerFast, T5Tokenizer, AutoTokenizer, BertForSequenceClassification, BertTokenizerFast,
    DistilBertForSequenceClassification, RobertaForSequenceClassification, T5ForConditionalGeneration,
    Trainer, TrainingArguments
    )
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset, concatenate_datasets
import logging
import sys
import copy
import os
from sklearn.metrics import accuracy_score
from datasets import concatenate_datasets
import logging
from scipy.stats import pearsonr
# from utils import *
from utils.split_data import DatasetSplitter,k_shot_data
from utils.adaptive import reordering_weights, gradient_masking_extraction, calculate_trainable_params

random_seed = 123

def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))

# set no_deprecation_warning to True to avoid warning messages
def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    if task == "stsb":
        pearson_corr, _ = pearsonr(predictions.squeeze(), labels)
        return {"pearson_corr": pearson_corr}
    else:
    
        predictions = predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    

def tokenize_function(examples, tokenizer, dataset):


    if dataset in ["sst2", "cola"]:

        return tokenizer(examples['sentence'], padding="max_length", truncation=True,return_tensors="pt")

    elif dataset == "mnli":
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True,return_tensors="pt")
    elif dataset == "qqp":
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True,return_tensors="pt")
    elif dataset == "qnli":
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True,return_tensors="pt")

    elif dataset in ["mrpc", "stsb", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True,return_tensors="pt")


def evaluate(args, global_model, tokenized_test_dataset):
    # tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)

    training_args = TrainingArguments(
       args.log_dir,
        logging_dir = args.log_dir,
        logging_steps = 1000,
        save_strategy = "no",
        evaluation_strategy="no",
    )

    global_model.to("cuda")  # Move the global model to GPU memory for evaluation
    # global_model = torch.compile(global_model)
    trainer = Trainer(
        model=global_model,
        args=training_args,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    true_labels = tokenized_test_dataset["label"]

    global_model.to("cpu")  # Move the global model back to CPU memory after evaluation

    if args.dataset == "stsb":
        pearson_corr = compute_metrics((predictions.predictions, true_labels), args.dataset)["pearson_corr"]
        print(f"Pearson correlation: {pearson_corr}")
        logging.info(f"Pearson correlation: {pearson_corr}")
        return pearson_corr
    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        logging.info(f"Accuracy: {accuracy}")
        return accuracy






def federated_learning(args, global_model, train_datasets, test_dataset, tokenizer):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    for communication_round in range(args.num_rounds):
        local_models = []
        lr = step_lr(args.learning_rate, communication_round, 5, 0.98)
        global_model.to('cpu')
        #randomly select 10% client index for training
        np.random.seed(int(time.time()))  # Set the seed to the current time
        client_indices = np.random.choice(len(train_datasets), size=int(0.1*len(train_datasets)), replace=False)
        avg_trainable_params = 0
        for idx, client_id in enumerate(client_indices):
        # for client_id, client_dataset in enumerate(train_datasets):
            client_dataset = train_datasets[client_id]
            print(f"Training client {client_id} in communication round {communication_round}")

            if idx == 0:
                local_model = copy.deepcopy(global_model)
                total_trainable_params,total_params, percentage = calculate_trainable_params(local_model)
            else:
                local_model,total_trainable_params, total_params, percentage = gradient_masking_extraction(global_model, target_model_params_size=None) #Target model params size is None for randomly sample subnetwork
            avg_trainable_params += total_trainable_params

            tokenized_client_dataset = client_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset), batched=True)


            writer.add_scalar(str(client_id) + '/trainable_params', total_trainable_params, communication_round)
            writer.add_scalar(str(client_id) + '/total_params', total_params, communication_round)
            print(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")     
            logging.info(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")

            writer.add_scalar(str(client_id) + '/trainable_params_percentage', percentage, communication_round)
            
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

            print("local model training finished")
            logging.info(f"local model training finished")
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

        writer.add_scalar("trainable_params/avg", avg_trainable_params/len(client_indices), communication_round)
        writer.add_scalar("trainable_params/org", total_params, communication_round)
        print(f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters")
        logging.info(f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters")

        res = evaluate(args, global_model, test_dataset, tokenizer)
        writer.add_scalar("test_accuracy", res, communication_round)
        print(f"Test accuracy is {res}")
        logging.info(f"Test accuracy is {res}")
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
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    
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
    elif args.model == "bert-base":
        global_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes[args.dataset])


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

#python fl_glue.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset sst2 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --model bert-base --log_dir glue/sst2 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=100)
    # parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_local_epochs", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "mrpc", "mnli", "qqp", "qnli", "stsb", "rte", "cola"], help="Choose between 'sst2', 'mrpc', 'mnli', 'qqp', 'qnli', 'stsb', 'rte', 'cola' datasets")
    parser.add_argument("--per_device_train_batch_size", type=int, default=40)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=40)
    parser.add_argument("--log_dir", type=str, default="centralized/4")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model", type=str, default="distilbert", choices=["distilbert", "bert-base","roberta", "t5"], help="Choose between 'distilbert', 'roberta', and 't5' models")
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
    writer = SummaryWriter(args.log_dir)
    main(args)