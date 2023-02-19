
import torch
from torch import nn
import logging
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Abstract base class for federated learning trainer.
    1. The goal of this abstract class is to be compatible to
    any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
    2. This class can be used in both server and client side
    3. This class is an operator which does not cache any states inside.
    """

    def __init__(self,logger=None) -> None:
        super().__init__()
        self.logger = logger

    @abstractmethod
    def train(self, model, dataloader , criterion, optimizer, local_epochs, device):
            """_summary_

            Args:
                model (nn.Module): _description_
                dataloader (_type_): _description_
                criterion (_type_): _description_
                optimizer (_type_): _description_
                epochs (int): _description_
                device (_type_): _description_
            """
            pass

    def aggregate(self, **kwargs):        
            """
            kwargs:
                nets_params: 
                local_datasize:
                global_para: 

            Returns:
                global_para: _description_
            """
            pass

    def test(self, model, test_data, device):

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
            "test_accuracy":0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        # if args.dataset == "stackoverflow_lr":
        #     criterion = nn.BCELoss(reduction="sum").to(device)
        # else:
            # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        metrics["test_accuracy"] = metrics["test_correct"] / len(test_data.dataset)
        return metrics

        