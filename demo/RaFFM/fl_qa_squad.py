import torch
import argparse
from transformers import (
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast, 
    T5Tokenizer, 
    AutoTokenizer, 
    RobertaForQuestionAnswering,
    )
from datasets import load_dataset
import logging
import sys
import copy
import os
from utils.split_data import DatasetSplitter, k_shot_data

# set no_deprecation_warning to True to avoid warning messages
import collections
from tqdm.auto import tqdm
import numpy as np
from transformers import BertForQuestionAnswering, DistilBertForQuestionAnswering
import evaluate
import time
from torch.utils.tensorboard import SummaryWriter
from utils.adaptive import reordering_weights, gradient_masking_extraction,calculate_trainable_params


random_seed = 123

def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))


def compute_metrics(start_logits, end_logits, features, examples):
    n_best = 20
    max_answer_length = 30

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    metric = evaluate.load("squad")
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def tokenize_function(examples,tokenizer):
    return tokenizer(examples['question'], examples['context'], truncation=True)

def prepare_train_features(examples,tokenizer):
    max_length = 384
    stride = 128
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples,tokenizer):
    max_length = 384
    stride = 128
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def prepare_train_features_squad_v2(examples,tokenizer):
    max_length = 384
    stride = 128
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    is_impossible = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            is_impossible.append(1)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
            is_impossible.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["is_impossible"] = is_impossible
    return inputs

def federated_learning(args, global_model, train_datasets, raw_datasets,tokenizer):
    # global_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    global_model.to('cpu')
    tokenized_client_datasets = []
    for client_dataset in train_datasets:
        tokenized_client_dataset = client_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
        # tokenized_client_dataset = [tokenize_function(example,tokenizer) for example in client_dataset[client_id]]

        tokenized_client_dataset = tokenized_client_dataset.map(
            lambda examples: prepare_train_features(examples, tokenizer),
            batched=True,
            remove_columns=raw_datasets['train'].column_names,
        )

        tokenized_client_datasets.append(tokenized_client_dataset)

    validation_dataset = raw_datasets["validation"].map(
        lambda examples: preprocess_validation_examples(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    for communication_round in range(args.num_rounds):

        lr = step_lr(args.learning_rate, communication_round, 5, 0.98)
        local_models = []
        global_model.to('cpu')
        #randomly select 10% client index for training
        np.random.seed(int(time.time()))  # Set the seed to the current time
        client_indices = np.random.choice(len(tokenized_client_datasets), size=int(0.1*len(tokenized_client_datasets)), replace=False)

        # for client_id, tokenized_client_dataset in enumerate(tokenized_client_datasets):
        avg_trainable_params = 0
        for client_id in client_indices:
            tokenized_client_dataset = tokenized_client_datasets[client_id]
            print(f"Training client {client_id} in communication round {communication_round}")
            # global_model = reordering_weights(global_model)
            local_model,total_trainable_params, total_params, percentage = gradient_masking_extraction(global_model, target_model_params_size=None) #Target model params size is None for randomly sample subnetwork
            avg_trainable_params += total_trainable_params
            

            writer.add_scalar(str(client_id) + '/trainable_params', total_trainable_params, communication_round)
            writer.add_scalar(str(client_id) + '/total_params', total_params, communication_round)
            print(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")     
            logging.info(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")

            writer.add_scalar(str(client_id) + '/trainable_params_percentage', percentage, communication_round)
            
            # local_model = copy.deepcopy(global_model)


            logdir = os.path.join(args.log_dir, f"client_{client_id}")

            training_args = TrainingArguments(
                logdir,
                # logging_dir = logdir,
                # logging_steps = 1000,
                # logging_strategy="epoch",
                evaluation_strategy="no",
                save_strategy = "no",
                learning_rate=lr,
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


            # predictions, _, _ = trainer.predict(validation_dataset)
            # start_logits, end_logits = predictions   
            # res = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
            # print("local model training finished, validation results: ", res)
            # #extract value from dict res
            # res = list(res.values())
            # logging.info(f"local model training finished, validation results: {res}")
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

        eval_trainer = Trainer(
            model=global_model,
            args=training_args,
            train_dataset=tokenized_client_dataset,
        )
        predictions, _, _ = eval_trainer.predict(validation_dataset)
        start_logits, end_logits = predictions   
        res = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        print("Global validation results: ", res)
        writer.add_scalar("F1/validation", res["f1"], communication_round)
        writer.add_scalar("EM/validation", res["exact_match"], communication_round)
        res = list(res.values())
        logging.info(f"Global validation results: {res}")
        
        
    return global_model


def main(args):
    if args.model == "distilbert":
        model_name = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        global_model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    elif args.model == "roberta":
        # raise NotImplementedError
        model_name = "roberta-base"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        global_model = RobertaForQuestionAnswering.from_pretrained(model_name)
    elif args.model == "t5":
        raise NotImplementedError
        model_name = "t5-small"  # You can also use "t5-base" or other T5 variants
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    elif args.model == "bert-base":
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        global_model = BertForQuestionAnswering.from_pretrained(model_name)
    elif args.model == "bert-large":
        model_name = 'bert-large-uncased-whole-word-masking'
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        global_model = BertForQuestionAnswering.from_pretrained(model_name)

    
    if args.dataset == "squad":
        datasets = load_dataset('squad')

    elif args.dataset == "squad_v2":
        datasets = load_dataset('squad_v2')
    
    else:
        raise NotImplementedError



    train_dataset = datasets["train"]
    # val_dataset = datasets["validation"]




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
    

    
    
    global_model = federated_learning(args, global_model,local_datasets, datasets,tokenizer)


    print(dash_line+"\nFinal evaluation")
    logging.info(dash_line+"\nFinal evaluation")
    # tokenize_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer, args.dataset), batched=True)
    # evaluate(args, global_model, tokenize_test_dataset)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_local_epochs", type=int, default=30)
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "squad_v2"], help="Choose between 'squad' and 'squad_v2' datasets")
    parser.add_argument("--per_device_train_batch_size", type=int, default=44)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=44)
    parser.add_argument("--log_dir", type=str, default="centralized/4")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model", type=str, default="bert-base", choices=["distilbert", "roberta", "t5", "bert-base", "bert-large"], help="Choose between 'distilbert', 'roberta', 't5', 'bert-base', 'bert-large'")
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

#python fl_qa_squad.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad --log_dir suqad/100 --model bert-base

# sbatch --gres=gpu:1 --wrap="python3 fl_qa_squad.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad_v2 --log_dir suqad/100 --model bert-base > squad/100/console.log"