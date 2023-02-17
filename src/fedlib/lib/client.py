from abc import ABC, abstractmethod
from http import client
import torch

class Client:
    def __init__(self, **kwargs) -> None:
        
        
        """        
        # TODO: private variables
        for key, val in kwargs:
            setattr(self, key, val)
        """

        self.id = kwargs["id"]
        self._model = kwargs["model"]
        self._trainloader = kwargs["trainloader"]
        self._testloader = kwargs["testloader"]
        self._lr = kwargs["lr"]

        self.datasize = len(self._trainloader.dataset)
        self._trainer = kwargs["trainer"]
        self._device = kwargs["device"]
        self._communicator = kwargs["communicator"]
        
        self._init_criterion(kwargs["criterion"])
        self._init_optimizer(kwargs["optimizer"])
        self._init_lr_schedular(kwargs["lr_scheduler"])

        # self._global_testloader = kwargs["test_dl_global"]
        

    def _init_optimizer(self,optimizer_name:str) -> None:
        """_summary_

        Args:
            optmizer_name (str): _description_
        """
        if optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self._model.parameters(), self._lr)
        else:
            raise KeyError("currently only support SGD")

    def _init_lr_schedular(self,lr_scheduler_name:str) -> None:
        """_summary_

        Args:
            lr_scheduler_name (str): _description_

        Raises:
            KeyError: _description_
        """
        if lr_scheduler_name.lower() == "exponentiallr": #ExponentialLR
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            raise KeyError

    def _init_criterion(self, criterion_name:str) -> None:
        """_summary_

        Args:
            criterion_name (str): _description_

        Raises:
            KeyError: _description_
            NotImplementedError: _description_
        """
        if not criterion_name:
            raise KeyError("Please Specify Criterion")
        elif criterion_name.lower() == 'CrossEntropyLoss'.lower():
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _communication(self):
        self._communicator.communication()
    
    def client_update(self,**kwargs):
        kwargs["dataloader"] = self._trainloader
        kwargs["device"] = self._device
        kwargs["model"] = self._model   
        kwargs["optimizer"] = self.optimizer
        kwargs["scheduler"] = self.scheduler
        kwargs["criterion"] = self.criterion

        self._trainer.train(**kwargs)
    
    @abstractmethod
    def client_run(self, **kwargs):
        pass
        # Some pseudo code idea:
        # self._communication()
        # # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

        # self.client_update(**kwargs)
        # self._communication(self)

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def get_model_params(self, module_name=None):
        
        if module_name is not None:
            try:
                return self._model.get_submodule(module_name).cpu().state_dict()
            except:
                raise KeyError("Module Not Exists")

        return self._model.cpu().state_dict()


    
    def set_model_params(self, model_parameters,module_name=None):
        
        if module_name is not None:
            try:
                self._model.get_submodule(module_name).load_state_dict(model_parameters)
            except:
                raise KeyError("Module Not Exists")
        
        else:
            self._model.load_state_dict(model_parameters)

    

    def get_dataset(self):
        return self._dataset
    
    def set_dataset(self, dataset):
        self._dataset = dataset

    def eval(self):
        return self._trainer.test(self._model, self._testloader, self._device)
    
    def eval_decoder(self):
        # TODO: integrate this to eval. In Trainer, write two function, test and test decoder, and 
        # use a new function to call this two functions, and use a parameter to control call which eval function
        return self._trainer.test_decoder(self._model, self._global_testloader, self._device)

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