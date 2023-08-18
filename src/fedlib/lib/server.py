from abc import abstractmethod
from fedlib.lib.sampler import *
from torch.utils.tensorboard import SummaryWriter
import os
class Server:
    def __init__(self,**kwargs) -> None:
        self._n_clients = kwargs["n_clients"]
        self._global_model = kwargs["global_model"]
        self._device = kwargs["device"]
        
        self._init_sampler(kwargs["sampler"])
        self._n_classes = kwargs["n_classes"]
        
        self._trainer = kwargs["trainer"]
        self._communicator = kwargs["communicator"]
        self._sim_matrix =  [[ 0 if i == j else None for j in range(self._n_clients)] for i in range(self._n_clients)]
        self._clusters = {}
        
        
        #TODO: change to testloader
        self._test_dataset = kwargs["test_dl_global"]
        self._val_dl_global = kwargs["val_dl_global"]
        
        #TODO: add attribute client participation rate
        
        if kwargs['log_dir']:
            self._init_writer(kwargs['log_dir'])

        '''initialize key pair'''
        self._key_generator()


    
    def _init_writer(self,log_dir):
        self.writer = SummaryWriter(log_dir)
    

    def _init_sampler(self,sampler_name:str) -> None:
        """_summary_

        Args:
            optmizer_name (str): _description_
        """
        if sampler_name.lower() == "random":
            self.sampler = random_sampler
        elif sampler_name.lower() == "stratified_cluster_sampler":
            self.sampler = stratified_cluster_sampler
        else:
            raise KeyError("currently only support random sampler")

    def _communication(self):
        if self._communicator:
            self._communicator.server()
    

    def _aggregate(self,**kwargs):
        global_params = self._trainer.aggregate(**kwargs)
        self.set_global_model_params(global_params)

    def server_update(self, **kwargs):
        self._aggregate(**kwargs)

    def eval(self,**kwargs):
        return self._trainer.test(model = self._global_model,test_data=self._test_dataset,device = self._device, **kwargs)

    @abstractmethod
    def server_run(self, test_gap=1):
        #Add a listen to receive sockets from clients
        pass
   
    def client_sample(self, **kwargs):
        return self.sampler(**kwargs)

    def class_probs(self):
        # Initialize the number of labels
        num_labels = self._n_classes

        # Initialize a dictionary to store class counts
        class_counts = {label: 0 for label in range(num_labels)}

        # Iterate over the dataset using the dataloader
        for inputs, labels in self._test_dataset:
            # Count the occurrences of each class
            for label in labels:
                class_counts[label.item()] += 1

        # Calculate the probabilities
        total_samples = len(self._test_dataset.dataset)
        class_probabilities = {
            label: count / total_samples for label, count in class_counts.items()
        }

        # Print the probabilities
        # for label, probability in class_probabilities.items():
        #     print(f"Class {label}: Probability {probability}")
        
        return class_probabilities

    def get_global_model(self):
        return self._global_model

    def set_global_model(self, model):
        self._global_model = model

    def get_global_model_params(self):
        return self._global_model.cpu().state_dict()

    def set_global_model_params(self, model_parameters):
        self._global_model.load_state_dict(model_parameters)

    def _validate(self):
        self._trainer.test()

    def save_ckpt(self):
        pass

    def load_ckpt(self):
        pass


    def get_dataset(self):
        return self._test_dataset
    
    def set_dataset(self, dataset):
        self._test_dataset = dataset

    def _server_encryption(self):
        pass

    
    def _server_decryption(self):
        pass


    def _key_generator(self):
        #self._public_key, self._private_key = None, None
        pass