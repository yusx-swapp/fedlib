from abc import ABC, abstractmethod
from http import client
import torch

class Client:
    def __init__(self, **kwargs) -> None:
        self.id = kwargs["id"]

        self._model = kwargs["model"]
        self._trainloader = kwargs["trainloader"]
        self._lr = kwargs["lr"]

        self.datasize = len(self._trainloader.dataset)

        self._trainer = kwargs["trainer"]
        self._device = kwargs["device"]
        self._communicator = kwargs["communicator"]
        if kwargs["optimizer"] == "SGD":
            self.optimizer = torch.optim.SGD(self._model.parameters(), self._lr)
        else:
            raise KeyError("currently only support SGD")
        
        if kwargs["lr_scheduler"] == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        else:
            raise KeyError
    def _communication(self):
        self._communicator.communication()
    
    def client_update(self,**kwargs):
        

        kwargs["dataloader"] = self._trainloader
        kwargs["device"] = self._device
        kwargs["model"] = self._model   
        kwargs["optimizer"] = self.optimizer           
        self._trainer.train(**kwargs)
    
    @abstractmethod
    def client_run(self, **kwargs):
        self._communication()
        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        self.client_update(**kwargs)
        self._communication(self)

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def get_model_params(self):
        return self._model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self._model.load_state_dict(model_parameters)
    
    def get_dataset(self):
        return self._dataset
    
    def set_dataset(self, dataset):
        self._dataset = dataset

    def eval(self):
        self._trainer.test()

    def save_ckpt(self):
        pass

    def load_ckpt(self):
        pass


    def _client_encryption(self):
        pass

    
    def _client_decryption(self):
        pass


    def _key_generator(self):
        #self._public_key, self._private_key = None, None
        pass

    def to(self, device):
        pass