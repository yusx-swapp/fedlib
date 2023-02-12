from fedlib.utils import Arguments
import torch
import numpy as np
import copy

from fedlib.ve import FEDDFEnv as Simulator
from fedlib.lib import Server, Client
from fedlib.networks import resnet20
from fedlib.lib.sampler import random_sampler
from fedlib.lib.algo import feddp
from fedlib.datasets import partition_data, get_dataloader,get_client_dataloader
from fedlib.utils import get_logger

def init_sampler(sampler_name = 'random'):
    if sampler_name == 'random':
        return random_sampler



def init_model(model_args):
    
    if model_args.model == 'resnet20':
        return resnet20()
    else:
        raise NotImplementedError

def init_server(server_args,global_model,trainer,test_dataset,communicator=None):
    
    server = Server(n_clients = server_args.n_clients, global_model= global_model,
                    device = server_args.device, sampler = server_args.sampler,
                    trainer = trainer,communicator = communicator,test_dataset=test_dataset)
    
    return server
def init_clients(client_args,model,data_loaders,testloader,trainer,communicator=None):
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


def main():
    logger = get_logger()
    args = Arguments()
    args.show_args()
    
    data_loaders, global_test_dl, _ = init_dataset(args.dataset)
    
    model = init_model(args.model)
    
    server = init_server(args.server,global_model=model,
                        trainer=feddp(logger),test_dataset=global_test_dl,
                        communicator=None)
    
    clients = init_clients(args.client,model=model,data_loaders=data_loaders,
                            testloader=global_test_dl,trainer=feddp(logger),
                            communicator=None)
    
    simulator = Simulator(server=server, clients=clients, 
                        communication_rounds=10,n_clients= 100,sample_rate=.1)
    
    simulator.run(local_epochs=1,pruning_threshold=1e-3)

if __name__ == '__main__':

    main()    