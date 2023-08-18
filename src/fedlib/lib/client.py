from abc import ABC, abstractmethod
import torch
import os
from torch.utils.tensorboard import SummaryWriter
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
        self._val_dl_global = kwargs["val_dl_global"]
        self._lr = kwargs["lr"]
        self._global_output = None
        self._n_classes = kwargs["n_classes"]

        self.datasize = len(self._trainloader.dataset)
        self._trainer = kwargs["trainer"]
        self._device = kwargs["device"]
        self._communicator = kwargs["communicator"]
        
        self._init_criterion(kwargs["criterion"])
        self._init_optimizer(kwargs["optimizer"])
        self._init_lr_schedular(kwargs["lr_scheduler"])
        if kwargs['log_dir']:
            self._init_writer(kwargs['log_dir'])

    
    def _init_writer(self,log_dir):
        self.writer = SummaryWriter(log_dir)

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
        self.criterion = torch.nn.CrossEntropyLoss()


    def _communication(self):
        self._communicator.communication()
    
    def class_probs(self):
        # Initialize the number of labels
        num_labels = self._n_classes

        # Initialize a dictionary to store class counts
        class_counts = {label: 0 for label in range(num_labels)}

        # Iterate over the dataset using the dataloader
        for inputs, labels in self._trainloader:
            # Count the occurrences of each class
            for label in labels:
                class_counts[label.item()] += 1

        # Calculate the probabilities
        total_samples = len(self._trainloader.dataset)
        class_probabilities = {
            label: count / total_samples for label, count in class_counts.items()
        }

        # Print the probabilities
        # for label, probability in class_probabilities.items():
        #     print(f"Class {label}: Probability {probability}")
        
        return class_probabilities
    
    def client_update(self,**kwargs):
        kwargs["dataloader"] = self._trainloader
        kwargs["device"] = self._device
        kwargs["model"] = self._model   
        kwargs["optimizer"] = self.optimizer
        kwargs["scheduler"] = self.scheduler
        kwargs["criterion"] = self.criterion

        self._trainer.train(**kwargs)
    
    def get_latent_vectors(self):
        return self._trainer.vectorize(self._model, self._val_dl_global, self._device)
        
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

    def get_model_params(self):
        return self._model.cpu().state_dict()

    
    def set_model_params(self, model_parameters):
        self._model.load_state_dict(model_parameters)
    

    def get_dataset(self):
        return self._dataset
    
    def set_dataset(self, dataset):
        self._dataset = dataset

    def eval(self):
        return self._trainer.test(self._model, self._testloader, self._device)

    def test_on_global(self):
        self._global_output = self._trainer.test_on_global(self._model,self._val_dl_global,self._device)
    
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