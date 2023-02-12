import copy
from typing import Dict

from ..datasets.prepare_data import partition_data, get_client_dataloader
from ..lib.sampler import *
from ..networks import *
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger
from ..lib.algo import feddp as trainer
class FEDDFEnv:
    def __init__(self, server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, participate_rate:float) -> None:
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
        self.participate_rate = participate_rate
        self.logger = get_logger()


    #Todo initialize by configuration
    def init_config(self) -> None:
        pass

    def run(self,local_epochs,pruning_threshold=1e-3):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.participate_rate)
            
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
            self.logger.info('*******Model Test Accuracy After Server Aggregation: %s *******' % str(metrics["test_accuracy"]))
            self.logger.info('*******Rounds %s Federated Learning Finished!******' % str(round+1))


def init_sampler(sampler_name = 'random'):
    if sampler_name == 'random':
        return random_sampler



def init_model(model_args):
    
    if model_args.model == 'resnet20':
        return resnet20()
    else:
        raise NotImplementedError

def init_server(server_args,global_model,test_dataset,trainer=trainer(get_logger()),communicator=None):
    
    server = Server(n_clients = server_args.n_clients, global_model= global_model,
                    device = server_args.device, sampler = server_args.sampler,
                    trainer = trainer,communicator = communicator,test_dataset=test_dataset)
    
    return server
def init_clients(client_args,model,data_loaders,testloader,trainer=trainer(get_logger()),communicator=None):
    clients = {}


    for id in range(client_args.n_clients):
        local_model = copy.deepcopy(model)
        clients[id] = Client(id=id,model=local_model,trainloader = data_loaders[id],
                            testloader=testloader,lr=client_args.lr,trainer=trainer,
                            device=client_args.device,communicator = communicator,
                            criterion=client_args.criterion,optimizer=client_args.optimizer,
                            lr_scheduler = client_args.lr_scheduler)

    return clients


def init_dataset(data_args):

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(data_args.dataset, data_args.datadir, \
                                                                                            data_args.partition, data_args.n_clients, \
                                                                                            data_args.beta)
    
    data_loaders, global_test_dl, test_loaders = get_client_dataloader(data_args.dataset, data_args.datadir, data_args.batch_size, data_args.n_worker, net_dataidx_map)

    return data_loaders, global_test_dl, test_loaders