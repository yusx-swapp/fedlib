from abc import abstractmethod
from torch import nn

class Server:
    def __init__(self,**kwargs):
        self._n_clients = kwargs["n_clients"]
        self._global_model = kwargs["global_model"]
        self._device = kwargs["device"]
        self._sample_fn = kwargs["sample_fn"]
        # self._aggregate_fn = kwargs["aggregate_fn"]
        self._trainer = kwargs["trainer"]
        self._communicator = kwargs["communicator"]

        self._test_dataset = kwargs["test_dl_global"]
        '''initialize key pair'''
        self._key_generator()
    
    def init(self, global_model: nn.Module, n_clients: int, device: str):
        """_summary_

        Args:
            global_model (nn.Module): _description_
            n_clients (int): _description_
            device (str): _description_
        """
        pass
    #ToDo communicator
    def _communication(self):
        self._communicator.communicate_fn()
        pass
    
    def _aggregate(self,**kwargs):
        global_params = self._trainer.aggregate(**kwargs)
        self.set_global_model_params(global_params)

    def server_update(self, **kwargs):
        self._aggregate(**kwargs)
            
    
    @abstractmethod
    def server_run(self, test_gap=1):
        for r in self.communication_rounds:
            local_updates_dic = self._communication()
            cloud_updates_dic = self._server_update(local_updates_dic)
            self._global_model = cloud_updates_dic["global_model"]
            
            if r % test_gap == 0:
                self._validate()

            self.save_ckpt()
            self._communication()
        pass
   
    def client_sample(self, **kwargs):
        return self._sample_fn(**kwargs)


    def get_global_model(self):
        return self._global_model

    def set_global_model(self, model):
        self._global_model = model
    
    def get_global_model_params(self):
        return self._global_model.cpu().state_dict()

    def set_global_model_params(self, model_parameters):
        self._global_model.load_state_dict(model_parameters)
    

    def eval(self):
        self._trainer.test(self._global_model,self._test_dataset, self._device)
    
    def eval_decoder(self):
        return self._trainer.test_decoder(self._global_model, self._test_dataset, self._device)

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

    def to(self,device):
        pass