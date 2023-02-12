from typing import Dict
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger

class FEDDFEnv:
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

    def run(self,local_epochs,pruning_threshold=1e-3):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.sample_rate)
            
            global_model_param = self.server.get_global_model_params()
            nets_params = []
            local_datasize = []
            self.logger.info('*******starting Rounds %s Optimization******' % str(round+1))
            self.logger.info('Participate Clients: %s' % str(selected))
            
            for id in selected:
                self.logger.info('Optimize the %s-th Clients' % str(id))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("id not match")
                
                client.set_model_params(global_model_param)
                client.client_update( epochs=local_epochs,pruning_threshold=pruning_threshold)
                
                nets_params.append(client.get_model_params())
                local_datasize.append(client.datasize)

                metrics = client.eval()
                self.logger.info(f'*******Client {str(id+1)} Training Finished! Test Accuracy: {str(metrics["test_accuracy"])} ******')


            self.server.server_update(nets_params=nets_params, local_datasize=local_datasize,global_model_param= global_model_param)
            metrics = self.server.eval()
            self.logger.info('Model Test Accuracy After Server Aggregation: %s ' % str(metrics["test_accuracy"]))
            self.logger.info('*******Rounds %s Federated Learning Finished!******' % str(round+1))


