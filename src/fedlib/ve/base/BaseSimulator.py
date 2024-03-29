from typing import Dict
from ...lib.server import Server
from ...lib.client import Client
from ...utils import get_logger
class BaseSimulator:
    def __init__(self, server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, sample_rate:float) -> None:
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
        
        self.n_clients = n_clients
        self.sample_rate = sample_rate
        self.logger = get_logger()


    #Todo initialize by configuration
    def init_config(self) -> None:
        pass

    def run(self,local_epochs):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.sample_rate)
            print(selected)
            global_model_param = self.server.get_global_model_params()
            nets_params = []
            local_datasize = []
            self.logger.info('*******starting rounds %s optimization******' % str(round+1))

            for id in selected:
                self.logger.info('optimize the %s-th clients' % str(id))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("id not match")
                
                client.set_model_params(global_model_param)
                client.client_update( epochs=local_epochs)
                
                nets_params.append(client.get_model_params())
                local_datasize.append(client.datasize)

            self.server.server_update(nets_params=nets_params, local_datasize=local_datasize,global_model_param= global_model_param)
            self.server.eval()


