
import torch
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
    def train(self, model, dataloader , criterion, optimizer, epochs, device):
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
        pass

        