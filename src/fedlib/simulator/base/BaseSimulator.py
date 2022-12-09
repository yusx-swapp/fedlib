import copy

import torch
from sklearn.metrics import confusion_matrix
from torch import optim
import numpy as np
import os
# from .. import compute_acc
from .. import get_dataloader
import torch.nn as nn
from cmath import inf
from copy import deepcopy
from typing import List
from ...lib.server import Server
from ...lib.client import Client
class simulator:
    def __init__(self, server: Server, clients: List[Client], communication_rounds:int) -> None:
        """_summary_

        Args:
            server (Server): _description_
            clients (List[Client]): _description_
            communication_rounds (int): _description_
        """
        super().__init__()
        self.server = server
        self.clients = clients
        self.communication_rounds = communication_rounds
        self.device = None
        self.optimizer = None
    def init_config(self) -> None:
        pass

    def run(self):
        for round in self.communication_rounds:
            global_model_param = self.server.get_global_model_params
            for client in self.clients:
                client.set_model_params(global_model_param)
                client.client_update(optimizer=self.optimizer, criterion = self.criterion, decive = self.device)
