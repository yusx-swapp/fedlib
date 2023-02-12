from abc import abstractmethod
from fedlib.lib.sampler import *

class Server:
    def __init__(self,**kwargs) -> None:
        self._n_clients = kwargs["n_clients"]
        self._global_model = kwargs["global_model"]
        self._device = kwargs["device"]
        
        self._init_sampler(kwargs["sampler"])
        
        self._trainer = kwargs["trainer"]
        self._communicator = kwargs["communicator"]

        self._test_dataset = kwargs["test_dataset"]
        

        '''initialize key pair'''
        self._key_generator()
    

    def _init_sampler(self,sampler_name:str) -> None:
        """_summary_

        Args:
            optmizer_name (str): _description_
        """
        if sampler_name.lower() == "random":
            self.sampler = random_sampler
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