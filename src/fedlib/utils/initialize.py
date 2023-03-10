import copy
from typing import Dict

from ..datasets.prepare_data import partition_data, get_client_dataloader

from ..networks import *
from ..lib.server import Server

from ..lib.client import Client

__all__=['init_model','init_server','init_clients','init_dataset']

def init_model(model_args):
    
    if model_args.model == 'resnet20':
        return resnet20(n_classes=model_args.n_classes)
    else:
        raise NotImplementedError

def init_server(n_clients,global_model,testloader,trainer,sampler,communicator=None,device=None,log_dir=None):

    server = Server(n_clients = n_clients, global_model= global_model,
                    device = device, sampler = sampler,
                    trainer = trainer,communicator = communicator,test_dataset=testloader, log_dir=log_dir)
    
    return server


def init_clients(n_clients,model,data_loaders,testloader,trainer, lr,lr_scheduler,criterion,optimizer,communicator=None,device=None,log_dir=None):

    clients = {}
    print("trainer:", trainer)
    if type(testloader) == list:#local test loader
        for id in range(n_clients):
            local_model = copy.deepcopy(model)

            clients[id] = Client(id=id,model=local_model,trainloader = data_loaders[id],
                                testloader=testloader[id],lr=lr,trainer=trainer,
                                device=device,communicator = communicator,
                                criterion=criterion,optimizer=optimizer,
                                lr_scheduler = lr_scheduler,log_dir=log_dir)
    else:
        for id in range(n_clients):
            local_model = copy.deepcopy(model)

            clients[id] = Client(id=id,model=local_model,trainloader = data_loaders[id],
                                testloader=testloader,lr=lr,trainer=trainer,
                                device=device,communicator = communicator,
                                criterion=criterion,optimizer=optimizer,
                                lr_scheduler = lr_scheduler,log_dir=log_dir)
    return clients

def init_dataset(local_test_loader = False, **data_args):

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset=data_args['dataset'], datadir=data_args['datadir'], \
                                                                                            partition=data_args['partition'],n_parties=data_args['n_clients'], \
                                                                                            beta=data_args['beta'])
    
    data_loaders, global_test_dl, test_loaders = get_client_dataloader(dataset=data_args['dataset'], datadir=data_args['datadir'], 
                                                                        train_bs= data_args['train_bs'], test_bs=data_args['test_bs'],n_worker = data_args['n_worker'], dataidxs=net_dataidx_map,local_test_loader = local_test_loader)

    return data_loaders, global_test_dl, test_loaders