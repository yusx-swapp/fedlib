import torch
import argparse
from transformers import (
    # Trainer,
    # TrainingArguments,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
     
    )
from transformers import Seq2SeqTrainingArguments as TrainingArguments
from transformers import Seq2SeqTrainer as Trainer
from datasets import load_dataset
import logging
import sys
import copy
import os
from utils.split_data import DatasetSplitter, k_shot_data
from utils import EarlyStopping
from functools import partial

# set no_deprecation_warning to True to avoid warning messages
import collections
from tqdm.auto import tqdm
import numpy as np
import evaluate
import time
from torch.utils.tensorboard import SummaryWriter
from utils.adaptive import calculate_trainable_params
from utils import salient_parameter_prioritization, salient_submodel_extraction
from functools import partial
# from sklearn.metrics import f1_score, accuracy_score
from transformers import EvalPrediction

def compute_exact_match(pred, label):
    return int(pred.strip() == label.strip())

def compute_f1(pred, label):
    pred_tokens = pred.strip().split()
    label_tokens = label.strip().split()

    common_tokens = set(pred_tokens) & set(label_tokens)

    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return int(pred_tokens == label_tokens)

    if len(common_tokens) == 0:
        return 0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(label_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_metrics(eval_pred: EvalPrediction, tokenizer, no_answer_string="no answer"):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    decoded_preds = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    # Compute overall metrics
    overall_f1 = sum(compute_f1(pred, label) for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
    overall_em = sum(compute_exact_match(pred, label) for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

    # Filter out examples where the ground truth label is "no answer"
    has_answer_preds = [pred for pred, label in zip(decoded_preds, decoded_labels) if label.strip() and label != no_answer_string]
    has_answer_labels = [label for label in decoded_labels if label.strip() and label != no_answer_string]

    # Compute metrics for has answer samples
    has_ans_f1 = sum(compute_f1(pred, label) for pred, label in zip(has_answer_preds, has_answer_labels)) / len(has_answer_preds) if has_answer_preds else 0
    has_ans_em = sum(compute_exact_match(pred, label) for pred, label in zip(has_answer_preds, has_answer_labels)) / len(has_answer_preds) if has_answer_preds else 0

    return {
        'F1': overall_f1,
        'ExactMatch': overall_em,
        'hasAnsF1': has_ans_f1,
        'hasAnsExactMatch': has_ans_em
    }



random_seed = 123

def step_lr(initial_lr, epoch, decay_step, decay_rate):
    return initial_lr * (decay_rate ** (epoch // decay_step))


def prepare_features_squad_(examples, tokenizer):
    max_length = 384
    pad_to_max_length = True
    filtered_examples = [example for example in examples if example['answers']['text']]

    # Format the input as T5 expects
    input_pairs = ["question: " + q + " context: " + c for q, c in zip(examples['question'], examples['context'])]
    inputs = tokenizer(input_pairs, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    
    # Tokenize the answers
    answer_texts = [answer['text'][0] for answer in examples['answers']]
    answers = tokenizer(answer_texts, padding="max_length", max_length=50, truncation=True, return_tensors="pt")

    # answers = tokenizer(examples['answers']['text'], padding="max_length", max_length=50, truncation=True, return_tensors="pt")

    # Return the formatted data
    return {
        'input_ids': inputs.input_ids, 
        'attention_mask': inputs.attention_mask, 
        'decoder_input_ids': answers.input_ids,
        'labels': answers.input_ids
    }
def prepare_features_squad(examples, tokenizer):
    max_length = 384
    max_answer_length = 50
    no_answer_string = "no answer"

    # Format the input as T5 expects
    input_pairs = ["question: " + q + " context: " + c for q, c in zip(examples['question'], examples['context'])]
    inputs = tokenizer(input_pairs, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    
    # Tokenize the answers; if no answer is present, use the predefined no_answer_string
    answer_texts = [answer['text'][0] if len(answer['text']) > 0 else no_answer_string for answer in examples['answers']]
    answers = tokenizer(answer_texts, padding="max_length", max_length=max_answer_length, truncation=True, return_tensors="pt")

    # Return the formatted data
    return {
        'input_ids': inputs.input_ids, 
        'attention_mask': inputs.attention_mask, 
        'decoder_input_ids': answers.input_ids,
        'labels': answers.input_ids
    }


def federated_learning(args, global_model, tokenized_local_datasets, raw_datasets,tokenizer):
    early_stopping = EarlyStopping(patience=10, verbose=True)

    global_model.to('cpu')
    tokenized_client_datasets = tokenized_local_datasets
    validation_dataset = raw_datasets["validation"].map(
        lambda examples: prepare_features_squad(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    
    best_model = copy.deepcopy(global_model.to('cpu'))
    best_acc = 0
    for communication_round in range(args.num_rounds):

        lr = step_lr(args.learning_rate, communication_round, 5, 0.98)
        local_models = []
        global_model.to('cpu')
        #randomly select 10% client index for training
        np.random.seed(int(time.time()))  # Set the seed to the current time
        client_indices = np.random.choice(len(tokenized_client_datasets), size=int(0.1*len(tokenized_client_datasets)), replace=False)
        
        if args.spp:
            global_model = salient_parameter_prioritization(global_model)

        # for client_id, tokenized_client_dataset in enumerate(tokenized_client_datasets):
        avg_trainable_params = 0
        
        for idx, client_id in enumerate(client_indices):
            tokenized_client_dataset = tokenized_client_datasets[client_id]
            print(f"Training client {client_id} in communication round {communication_round}")
            # global_model = reordering_weights(global_model)
            

            if args.algo == 'raffm':

                if idx == 0:
                    local_model = copy.deepcopy(global_model)
                    total_trainable_params,total_params, percentage = calculate_trainable_params(local_model)
                    
                else:
                    local_model,total_trainable_params, total_params, percentage = salient_submodel_extraction(global_model, target_model_params_size=None) #Target model params size is None for randomly sample subnetwork
            elif args.algo == 'vanilla':
                local_model = copy.deepcopy(global_model)
                total_trainable_params,total_params, percentage = calculate_trainable_params(local_model)
            avg_trainable_params += total_trainable_params
            


            print(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")     
            logging.info(f"Client {client_id} has {total_trainable_params} trainable parameters out of {total_params} parameters, which is {percentage}% in communication round {communication_round}")

            
            # local_model = copy.deepcopy(global_model)


            logdir = os.path.join(args.log_dir, f"client_{client_id}")

            training_args = TrainingArguments(
                logdir,
                # logging_dir = logdir,
                # logging_steps = 1000,
                # logging_strategy="epoch",
                evaluation_strategy="no",
                save_strategy = "no",
                predict_with_generate=True,
                learning_rate=lr,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                num_train_epochs=args.num_local_epochs,
                weight_decay=0.01,
            )
            eval_dataset = validation_dataset.select(range(1000))  # evaluate on the first 1000 examples

            trainer = Trainer(
                model=local_model,
                args=training_args,
                train_dataset=tokenized_client_dataset,
                # eval_dataset = eval_dataset,
                # compute_metrics=partial(compute_metrics,tokenizer=tokenizer)
            )

            trainer.train()
            local_model.eval()
            predictions = trainer.predict(eval_dataset)

            res = compute_metrics(predictions,tokenizer=tokenizer)
            print("local model training finished, validation results: ", res)
            
            #extract value from dict res
            res = list(res.values())
            logging.info(f"local model training finished, validation results: {res}")
            
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
        print(f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters")
        logging.info(f"Average trainable parameters is {avg_trainable_params/len(client_indices)} out of {total_params} parameters")

        eval_trainer = Trainer(
            model=global_model,
            args=training_args,
            train_dataset=tokenized_client_dataset,
        )
        global_model.eval()
        predictions= eval_trainer.predict(validation_dataset)
        res = compute_metrics(predictions,tokenizer)
 

        print("Global validation results: ", res)
        res = list(res.values())
        logging.info(f"Global validation results: {res}")
        
        if res[1] > best_acc:
            best_acc = res[1]
            best_model = copy.deepcopy(global_model.to('cpu'))
        early_stopping(res[1])
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return global_model,best_model


def main(args):
    if args.model == 't5-base':
        model_name='google/flan-t5-base'
    elif args.model =='t5':
        model_name='google/flan-t5-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    global_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    

    

    datasets = load_dataset('squad_v2')




    train_dataset = datasets["train"]
    # val_dataset = datasets["validation"]



    tokenized_train_dataset = train_dataset.map(lambda examples: prepare_features_squad(examples, tokenizer), batched=True)

    dash_line = "-" * 80

    if args.k_shot and not args.split_data:
        print(dash_line+"\nFederated learning for few-shot learning")
        logging.info(dash_line+"\nFederated learning for few-shot learning")
        tokenized_local_datasets = k_shot_data(tokenized_train_dataset, args.num_clients, args.k_shot,args.dataset)
    else:
        print(dash_line+"\nFederated learning")
        logging.info(dash_line+"\nFederated learning")
        splitter = DatasetSplitter(tokenized_train_dataset, seed=random_seed)

        tokenized_local_datasets = splitter.split(n=args.num_clients, replacement=False)
    

    
    
    global_model,best_model = federated_learning(args, global_model,tokenized_local_datasets, datasets,tokenizer)

    print(dash_line+"\nTraining finished")
    logging.info(dash_line+"\nTraining finished")

        
    if args.save_model:
        print(dash_line+"\nFinal Save The Best Model")
        logging.info(dash_line+"\nFinal Save The Best Model")
        best_model.save_pretrained(os.path.join(args.log_dir, "best_model"))
        print(dash_line+"\nBest Model Saved")
        logging.info(dash_line+"\nBest Model Saved")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--method", choices=["centralized", "federated", "federated_foundation"], required=True)
    parser.add_argument("--algo", type=str, default='raffm', choices=['vanilla','raffm'])
    parser.add_argument("--spp", action="store_true", help="salient parameter prioritization")
    parser.add_argument("--save_model", action="store_true")
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
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--model", type=str, default="t5", choices=["t5",'t5-large','t5-base'], help="Choose between 'distilbert', 'roberta', 't5', 'bert-base', 'bert-large'")
    parser.add_argument("--eval_lm", action="store_true", help="evaluate local models")

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

#CUDA_VISIBLE_DEVICES=2 python fl_t5_squadv2.py --algo vanilla --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --dataset squad_v2 --log_dir log_t5 --model t5 > baseline_t5small_squadv2_100.txt
#CUDA_VISIBLE_DEVICES=1 python fl_t5_squadv2.py --algo vanilla --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --dataset squad_v2 --log_dir log_t5 --model t5-base > baseline_t5base_squadv2_100.txt

#python fl_t5_squadv2.py --model t5 --spp --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --dataset squad_v2 --log_dir log_small > raffm_t5small_squadv2_100.txt
#python fl_t5_squadv2.py --model t5-base --spp --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --dataset squad_v2 --log_dir log_base> raffm_t5base_squadv2_100.txt

#python fl_qa_squad.py --save_model --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --dataset squad --log_dir log_squad_bert_large --model bert-large > raffm_squad_bertlarge_100.txt



# sbatch --gres=gpu:1 --wrap="python3 fl_qa_squad.py --split_data --num_clients 100 --num_rounds 100 --num_local_epochs 3 --dataset squad_v2 --log_dir suqad/100 --model bert-base > squad/100/console.log"