from abc import abstractmethod
from typing import Dict
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger
__all__ = ['base']
class base:
    
    
    def __init__(self,server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, participate_rate:float) -> None:
        """_summary_

        Args:
            server (Server): _description_
            clients (List[Client]): _description_
            communication_rounds (int): _description_
        """
        self.server = server
        self.clients = clients
        self.communication_rounds = communication_rounds
        
        self.n_clients = n_clients
        self.participate_rate = participate_rate
        self.logger = get_logger()


    @abstractmethod
    def run(self,**trainer_args):
        pass

