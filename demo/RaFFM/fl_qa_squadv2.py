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
from split_data import DatasetSplitter, k_shot_data

# set no_deprecation_warning to True to avoid warning messages
import collections
from tqdm.auto import tqdm
import numpy as np
from transformers import BertForQuestionAnswering, DistilBertForQuestionAnswering
import evaluate
import time

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
            # If no valid answer, predict 'no_answer'
        if len(answers) == 0:
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": "",
                "no_answer_probability": 1.0  # Assuming you're certain there's no answer
            })
        else:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({
                "id": example_id, 
                "prediction_text": best_answer["text"],
                "no_answer_probability": 0.0  # Assuming you're certain there is an answer
            })
        # # Select the answer with the best score
        # if len(answers) > 0:
        #     best_answer = max(answers, key=lambda x: x["logit_score"])
        #     predicted_answers.append(
        #         {"id": example_id, "prediction_text": best_answer["text"]}
        #     )
        # else:
        #     predicted_answers.append({"id": example_id, "prediction_text": ""})
    metric = evaluate.load("squad_v2")
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def tokenize_function(examples,tokenizer):
    return tokenizer(examples['question'], examples['context'], truncation=True)

def prepare_train_features(examples,tokenizer):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def preprocess_validation_examples(examples,tokenizer):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


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
        for client_id in client_indices:
            tokenized_client_dataset = tokenized_client_datasets[client_id]
            print(f"Training client {client_id} in communication round {communication_round}")

            local_model = copy.deepcopy(global_model)


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
        
        eval_trainer = Trainer(
            model=global_model,
            args=training_args,
            train_dataset=tokenized_client_dataset,
            # compute_metrics=compute_metrics1,
        )
        predictions, _, _ = eval_trainer.predict(validation_dataset)
        start_logits, end_logits = predictions   
        res = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        print("Global validation results: ", res)
        logging.info(f"Global validation results: {str(res)}")
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

    

    datasets = load_dataset('squad_v2')
    



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

    main(args)

#python fl_qa_squad.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad --log_dir suqad/100 --model bert-base

# sbatch --gres=gpu:1 --wrap="python3 fl_qa_squadv2.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad_v2 --log_dir suqadv2/100 --model bert-base --per_device_train_batch_size 16 --per_device_eval_batch_size 16"