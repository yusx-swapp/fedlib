from typing import Dict
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger
from .base import base

__all__ = ['simulator']
class simulator(base):
    """Federated Learning Sequential Simulator.
        Use a for loop to simulate the federated learning process.
    """
    
    def __init__(self,server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, participate_rate:float) -> None:
        """

        Args:
            server (Server): _description_
            clients (List[Client]): _description_
            communication_rounds (int): _description_
        """
        # super.__init__()
        self.server = server
        self.clients = clients
        self.communication_rounds = communication_rounds
        
        self.n_clients = n_clients
        self.participate_rate = participate_rate
        self.logger = get_logger()


    def run(self,**trainer_args):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.participate_rate)
            
            global_model_param = self.server.get_global_model_params()
            nets_params = []
            local_datasize = []
            self.logger.info('*******starting Rounds %s Optimization******' % str(round+1))
            self.logger.info('Participate Clients: %s' % str(selected+1))
            avg_acc = 0
            for id in selected:
                self.logger.info('Optimize the %s-th Clients' % str(id+1))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("id not match")
                
                client.set_model_params(global_model_param)
                client.client_update(**trainer_args)
                
                nets_params.append(client.get_model_params())
                local_datasize.append(client.datasize)

                metrics = client.eval()
                self.logger.info(f'*******Client {str(id+1)} Training Finished! Test Accuracy: {str(metrics["test_accuracy"])} ******')
                client.writer.add_scalar(f'Accuracy/Client {str(id+1)}', metrics["test_accuracy"],round+1)
                avg_acc+=metrics["test_accuracy"]
            avg_acc = avg_acc/(len(selected))

            self.server.server_update(nets_params=nets_params, local_datasize=local_datasize,global_model_param= global_model_param)
            metrics = self.server.eval()
            self.logger.info('*******Model Test Accuracy After Server Aggregation: %s *******' % str(metrics["test_accuracy"]))
            self.server.writer.add_scalar('Accuracy/Global', metrics["test_accuracy"], round+1)
            self.logger.info('*******Model Avg Local Accuracy: %s *******' % str(avg_acc))
            self.server.writer.add_scalar('Accuracy/Avg Local Acc', avg_acc, round+1)
            self.logger.info('*******Rounds %s Federated Learning Finished!******' % str(round+1))


