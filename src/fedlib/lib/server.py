from abc import abstractmethod


class Server:
    def __init__(self,**kwargs) -> None:
        self._n_clients = kwargs["n_clients"]
        self._global_model = kwargs["global_model"]
        self._device = kwargs["device"]
        self._sample_fn = kwargs["sample_fn"]
        # self._aggregate_fn = kwargs["aggregate_fn"]
        self._trainer = kwargs["trainer"]
        self._communicator = kwargs["communicator"]

        self._test_dataset = kwargs["test_dataset"]
        '''initialize key pair'''
        self._key_generator()
    
    def _communication(self):
        if self._communicator:
            self._communicator.server()
    
    def _aggregate(self,**kwargs):
        self._trainer.aggregate(kwargs)

    def _server_update(self, **kwargs):
        self._aggregate(self.communication())
    
    @abstractmethod
    def server_run(self, test_gap=1):
        #Add a listen to receive sockets from clients
        pass
   
    def client_sample(self, **kwargs):
        self._sample_fn(**kwargs)


    def get_global_model(self):
        return self._global_model

    def set_global_model(self, model):
        self._global_model = model


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