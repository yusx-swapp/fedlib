from torch.utils.data import Subset
import numpy as np
import random

class DatasetSplitter:
    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        if seed is not None:
            random.seed(seed)

    def split(self, n, replacement=False):
        if replacement:
            return self._split_with_replacement(n)
        else:
            return self._split_without_replacement(n)

    def _split_with_replacement(self, n):
        size = len(self.dataset) // n
        sub_datasets = []
        for _ in range(n):
            indices = random.choices(range(len(self.dataset)), k=size)
            sub_dataset = [self.dataset[i] for i in indices]
            sub_datasets.append(sub_dataset)
        return sub_datasets

    def _split_without_replacement(self, n):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        size = len(indices) // n
        sub_datasets = [indices[i*size:(i+1)*size] for i in range(n)]
        # convert indices to actual dataset elements
        sub_datasets = [[self.dataset[i] for i in sub_dataset] for sub_dataset in sub_datasets]
        # add remaining elements to the last sub-dataset
        if len(indices) % n != 0:
            sub_datasets[-1].extend(self.dataset[i] for i in indices[n*size:])
        return sub_datasets


if __name__ == '__main__':
    from datasets import load_dataset
    dataset_name = 'sst2'
    dataset = load_dataset("glue", dataset_name)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

        #split the test set to local datasets for clients
    splitter = DatasetSplitter(train_dataset, seed=123)

    local_datasets = splitter.split(n=10, replacement=False)
