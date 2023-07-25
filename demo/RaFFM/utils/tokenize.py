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
