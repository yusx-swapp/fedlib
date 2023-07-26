import torch
import argparse
from sklearn.metrics import accuracy_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers import RobertaTokenizerFast, T5Tokenizer
from transformers import DistilBertForSequenceClassification, RobertaForSequenceClassification, T5ForConditionalGeneration
import numpy as np
from datasets import load_dataset, concatenate_datasets, load_from_disk
import logging
import sys
import copy
import os
from scipy.stats import pearsonr

# set no_deprecation_warning to True to avoid warning messages





def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred
    if task == "stsb":
        pearson_corr, _ = pearsonr(predictions.squeeze(), labels)
        return {"pearson_corr": pearson_corr}
    else:
    
        predictions = predictions.argmax(-1)
        return {"accuracy": accuracy_score(labels, predictions)}

"""
def tokenize_function2(examples, tokenizer, dataset,model=None):
    if dataset in ["sst2", "cola"]:

        return tokenizer(examples['sentence'], padding="max_length", truncation=True)

    elif dataset == "mnli":
        return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True)
    elif dataset == "qqp":
        return tokenizer(examples["question1"], examples["question2"], padding="max_length", truncation=True)
    elif dataset == "qnli":
        return tokenizer(examples["question"], examples["sentence"], padding="max_length", truncation=True)

    elif dataset in ["mrpc", "stsb", "rte"]:
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
"""

def tokenize_function(examples, tokenizer, dataset,model=None):


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



def evaluate(args, global_model, test_dataset, tokenizer):
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)

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
    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        logging.info(f"Accuracy: {accuracy}")

def centralized_few_shot_learning(args, global_model, train_dataset, test_dataset, tokenizer):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    global_model = global_model.cpu()
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)
    logging.info("=====> train_dataset size: {}".format(len(tokenized_train_dataset)))
    print("=====> train_dataset size: {}".format(len(tokenized_train_dataset)))
    training_args = TrainingArguments(
        args.log_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit = 1,
        save_strategy = "no",
        evaluation_strategy="epoch",
        # load_best_model_at_end=True,
        
    )

    training_args = TrainingArguments(
    output_dir=args.log_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit = 1,
    save_strategy = "no",
    evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=global_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,  # Pass tokenized_test_dataset instead of test_dataset
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, args.dataset),
    )


    #get the best model
    

    trainer.train()
    global_model = trainer.model
    return global_model

def federated_learning(args, global_model, train_datasets, test_dataset, tokenizer):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    for communication_round in range(args.num_rounds):
        local_models = []

        for client_id, client_dataset in enumerate(train_datasets):
            print(f"Training client {client_id} in communication round {communication_round}")

            local_model = copy.deepcopy(global_model)

            tokenized_client_dataset = client_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset, args.model), batched=True)

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

def federated_foundation(args, global_model,train_datasets, test_dataset, tokenizer):
    for communication_round in range(args.num_rounds):
        print("=====> Global model optimization on the centralized publi few-shot dataset")
        centralized_few_shot_dataset = train_datasets[-1]  # Use the last k-shot dataset
        global_model = centralized_few_shot_learning(args, global_model, centralized_few_shot_dataset, test_dataset, tokenizer)
        evaluate(args, global_model, test_dataset, tokenizer)
        global_model.to("cpu")  # Move the global model to CPU memory
        print("=====> Training the global model with federated learning")

        local_models = []

        for client_id, client_dataset in enumerate(train_datasets):
            print(f"Training client {client_id} in communication round {communication_round}")

            local_model = copy.deepcopy(global_model)
            # local_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
            # local_model.load_state_dict(global_model.state_dict())
            # local_model = torch.compile(local_model)
            tokenized_client_dataset = client_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset,args.model), batched=True)

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


        print("=====> Federated learning finished")
        logging.info("=====> Federated learning finished")
        print("=====> Evaluating global model after FL training finished")
        logging.info("=====> Evaluating global model after FL training finished")

        # evaluate(args, global_model, test_dataset, tokenizer)


        print(f"=====>Evaluating global model after communication round {communication_round}")
        logging.info(f"=====>Evaluating global model after communication round {communication_round}")
        evaluate(args, global_model, test_dataset, tokenizer)

    


    
  
    return global_model


def split_dataset(dataset, num_clients, k_shot, dataset_name):
    datasets = []

    if dataset_name in ["sst2", "mrpc", "qnli", "mnli", "qqp", "rte", "cola"]:
        class_examples = []
        num_classes = 3 if dataset_name == "mnli" else 2

        for i in range(num_classes):
            class_examples.append(dataset.filter(lambda example: example['label'] == i).shuffle())

        examples_per_client = k_shot * num_classes

        for i in range(num_clients):
            subsets = []

            for j in range(num_classes):
                start = i * k_shot
                end = (i + 1) * k_shot
                subsets.append(class_examples[j].select(range(start, end)))

            client_dataset = concatenate_datasets(subsets)
            datasets.append(client_dataset)

    elif dataset_name == "stsb":
        dataset = dataset.shuffle()
        examples_per_client = k_shot * 2  # Assuming an equal number of examples for high and low scores

        for i in range(num_clients):
            start = i * examples_per_client
            end = (i + 1) * examples_per_client

            client_dataset = dataset.select(range(start, end))
            datasets.append(client_dataset)

    return datasets



def main(args):
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
    elif args.model == "roberta":
        model_name = "roberta-base"
    elif args.model == "t5":
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
    

    if args.model == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    elif args.model == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    elif args.model == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    
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
    test_dataset = dataset["validation"]



    if args.method == "centralized":
        train_datasets = split_dataset(train_dataset, 1, args.k_shot,args.dataset)
        print("Centralized few-shot learning")
        logging.info("Centralized few-shot learning")
        train_datasets = train_datasets[0]
        global_model = centralized_few_shot_learning(args, global_model,train_datasets, test_dataset, tokenizer)


    elif args.method == "federated":
        train_datasets = split_dataset(train_dataset, args.num_clients, args.k_shot,args.dataset)
        print("Federated learning for few-shot learning")
        logging.info("Federated learning for few-shot learning")
        global_model = federated_learning(args, global_model,train_datasets, test_dataset, tokenizer)


    elif args.method == "federated_foundation":
        print("Federated Foundation Model Training for Few-Shot Learning")
        logging.info("Federated Foundation Model Training for Few-Shot Learning")
        train_datasets = split_dataset(train_dataset, args.num_clients+1, args.k_shot,args.dataset)

        global_model = federated_foundation(args, global_model,train_datasets, test_dataset, tokenizer)

    print("Final evaluation")
    logging.info("Final evaluation")
    evaluate(args, global_model, test_dataset, tokenizer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
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


#python ffm.py --method centralized --model t5 --dataset sst2 --k_shot 4 --num_epochs 60 --log_dir t5_centralized/16 >  t5/sst2_16.txt

#python ffm.py --method centralized --model roberta --dataset sst2 --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/4 >  roberta/sst2_16.txt
#python ffm.py --method federated --model roberta --dataset sst2 --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/4 >  roberta/sst2_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset sst2 --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/4 >  roberta/sst2_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset mrpc --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/mrpc_16.txt
#python ffm.py --method federated --model roberta --dataset mrpc --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/mrpc_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset mrpc --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/mrpc_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset mnli --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/mnli_16.txt
#python ffm.py --method federated --model roberta --dataset mnli --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/mnli_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset mnli --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/mnli_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset qqp --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/qqp_16.txt
#python ffm.py --method federated --model roberta --dataset qqp --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/qqp_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset qqp --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/qqp_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset qnli --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/qnli_16.txt
#python ffm.py --method federated --model roberta --dataset qnli --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/qnli_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset qnli --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/qnli_federated_foundation_16.txt


#python ffm.py --method centralized --model roberta --dataset stsb --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/stsb_16.txt
#python ffm.py --method federated --model roberta --dataset stsb --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/stsb_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset stsb --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/stsb_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset rte --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/rte_16.txt
#python ffm.py --method federated --model roberta --dataset rte --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/rte_federated_16.txt
#python ffm.py --method federated_foundation --model roberta --dataset rte --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/rte_federated_foundation_16.txt

#python ffm.py --method centralized --model roberta --dataset cola --k_shot 16 --num_epochs 60 --log_dir roberta_centralized/16 >  roberta/cola_16.txt
#python ffm.py --method federated --model roberta --dataset cola --k_shot 16 --num_local_epochs 30 --num_epochs=30  --num_rounds 3 --log_dir roberta_federated/16 >  roberta/cola_federated_16.txt

#python ffm.py --method federated_foundation --model roberta --dataset cola --k_shot 16 --num_local_epochs 30 --num_epochs=30 --num_rounds 3 --log_dir roberta_federated_foundation/16 >  roberta/cola_federated_foundation_16.txt