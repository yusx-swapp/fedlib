from abc import abstractmethod
from typing import TypeVar
from ...client import Client
class BaseCommunicator:

    def __init__(self) -> None:
        self.clients_info = []
        self.server_info = []
    
    @abstractmethod
    def client_communication():
        pass
    
    @abstractmethod
    def server_communication():
        pass

    @abstractmethod
    def client_encryption():
        pass

    @abstractmethod
    def client_decryption():
        pass

    @abstractmethod
    def server_encryption():
        pass

    @abstractmethod
    def server_decryption():
        pass

    @abstractmethod
    def key_generation():
        pass

