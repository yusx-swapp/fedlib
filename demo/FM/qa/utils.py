from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets
import logging
from scipy.stats import pearsonr

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
    else:
        predicted_labels = predictions.predictions.argmax(-1)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy}")
        logging.info(f"Accuracy: {accuracy}")




def k_shot_data(dataset, num_clients, k_shot, dataset_name):
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





def evaluate_old(args, global_model, test_dataset, tokenizer):
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
