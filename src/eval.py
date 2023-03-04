
import torch

import datetime

import torch
import numpy as np
import copy
import argparse
from fedlib.utils import get_logger, init_logs
from fedlib.ve.mtfl import MTFLEnv
from fedlib.lib import Server, Client
from fedlib.networks import resnet20
from fedlib.lib.sampler import random_sampler
from fedlib.lib.algo.torch.mtfl import Trainer
from fedlib.datasets import partition_data, get_dataloader,get_client_dataloader
from fedlib.networks import VAE

from torch import nn
from torch.utils.data import Subset

import torch
import torch.nn as nn

class NISTAutoencoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(NISTAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*4*4, embedding_dim),
        )
        #Predictor
        self.predictor = nn.Linear(in_features=embedding_dim, out_features=10, bias=True)
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1), # Transpose convolutional layer 1,
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def decoder_forward(self,x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        
        return x_
    
    def predictor_forward(self,x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        pred = self.predictor(z)
        
        return pred


    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        #z = z.view(z.size(0), -1)
        pred = self.predictor(z)
        
        return pred, x_


# class NISTAutoencoder(nn.Module):
#     def __init__(self):
#         super(NISTAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.predictor = nn.Linear(in_features=32, out_features=10, bias=True)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )
    
#     def decoder_forward(self,x):
#         z = self.encoder(x)
#         x_ = self.decoder(z)
        
#         return x_
    
#     def predictor_forward(self,x):
#         z = self.encoder(x)
#         z = z.view(z.size(0), -1)
#         pred = self.predictor(z)
        
#         return pred


#     def forward(self, x):
#         z = self.encoder(x)
#         x_ = self.decoder(z)
#         z = z.view(z.size(0), -1)
#         pred = self.predictor(z)
        
#         return pred, x_

class Cifar10Autoencoder(nn.Module):
    def __init__(self):
        super(Cifar10Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.predictor = nn.Sequential(
            nn.Linear(in_features=9216, out_features=120, bias=True),
            nn.Linear(in_features=120, out_features=84, bias=True),
            nn.Linear(in_features=84, out_features=10, bias=True)
        )
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        pred = self.predictor(x)
        x_ = self.decoder(x)
        return pred, x_

class Cifar100Autoencoder(nn.Module):
    def __init__(self):
        super(Cifar100Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.predictor = nn.Sequential(
            nn.Linear(in_features=9216, out_features=120, bias=True),
            nn.Linear(in_features=120, out_features=84, bias=True),
            nn.Linear(in_features=84, out_features=100, bias=True)
        )
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        pred = self.predictor(x)
        x_ = self.decoder(x)
        return pred, x_

def cluster(net_dataidx_map,y_train,n_clients):
    """Construct task clusters based on the nodes data partition.
    Each cluster will contain nodes that own the same set of data labels.
    Each node's task is to classify only the labels it possesses.
    This way, nodes in a cluster will have the same downstream task.

    Args:
        net_dataidx_map (dict): Data partition dictionary.
        y_train (np.array): All client labels.
        n_clients (int): Number of clients.

    Returns:
        tuple: Task:Nodes dictionary, Node:Task dictionary
    """
    task_nodes, node_task = {},{}
    for id in range(n_clients):
        idxs = net_dataidx_map[id]
        labels = np.unique(np.array(y_train)[idxs])
        node_task[id] = tuple(labels)
        if tuple(labels) in task_nodes.keys():
            task_nodes[tuple(labels)].append(id)
        else:
            task_nodes[tuple(labels)] = [id]

    # for task,nodes in task_nodes.items():
    #     print("\t",task, ":", nodes)
    
    return task_nodes, node_task


def customize_client_model(model,y_train,net_dataidx_map,id):
    """Customize client models so that the predictor layer will have
    as many neurons as the number of distinct labels assigned to the
    node in net_dataindex_map. Note that the output layer size will fluctuate 
    due to the non-iid data partition. Return a label map with key: real_label,
    value: new_label.

    Args:
        model (torch.nn): The client model.
        y_train (np.array): All client labels.
        net_dataidx_map (dict): Data partition dictionary.
        id (int): Client id.

    Returns:
        dict: Mapping of real_label to new_label.
    """
    idxs = net_dataidx_map[id]
    new_labels = np.unique(np.array(y_train)[idxs])
    out_features = len(new_labels)
    label_map = {j:i for i,j in enumerate(new_labels)}
    
    in_features = model.predictor.in_features
    model.predictor = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    # print("Client",id)
    # print("\t",label_map)
    # print("\t",model.predictor)

    return label_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='res18AE', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=20, help='number of local epochs')
    parser.add_argument('--pre_epochs', type=int, default=10, help='number of local pre train epochs')
    parser.add_argument('--n_clients', type=int, default=100,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--pretrained', type=int, default=0, help='Load pretrained model')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--local_acc', type=int, default=1, help='Enable local accuracy collection [0,1]')
    parser.add_argument('--sample', type=float, default=0.1, help='Sample ratio for each communication round')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    
    NOW = str(datetime.datetime.now()).replace(" ","--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = './logs/{}_dataset[{}]_model[{}]_partition[{}]_nodes[{}]_rounds[{}]_frac[{}]_local_ep[{}]_local_bs[{}]_beta[{}]_noise[{}]/'. \
        format(NOW,args.dataset, args.model, args.partition, args.n_clients, args.comm_round, args.sample,args.epochs, args.batch_size, args.beta, args.noise)

    args = vars(args)
    
    init_logs(log_file_name, args, log_dir)
    logger = get_logger()

    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['sample_fn'] = random_sampler
    args['trainer'] = Trainer(logger)
    args['communicator'] = None
    args["datadir"] = "./data"
    args["lr_scheduler"] = "ExponentialLR"  

    print("Args:",args)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    args["dataset"], args["datadir"], args['partition'], args['n_clients'], beta=args['beta'])
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args["dataset"],
                                                                                    args["datadir"],
                                                                                      args["batch_size"],
                                                                                      32)
    print("train_dl_global:",len(train_dl_global.dataset))
    print("test_dl_global:",len(test_dl_global.dataset))
    
    args["test_dl_global"] = test_dl_global

    if args["dataset"] in ["mnist","fmnist","femnist"]:
        model = NISTAutoencoder()
        x = torch.rand([10,1,28,28])
    elif args["dataset"] == "cifar10":
        model = VAE(1000,10)
        #model = Cifar10Autoencoder()
        x = torch.rand([10,3,32,32])
    elif args["dataset"] == "cifar100":
        model = VAE(1000,100)
        #model = Cifar100Autoencoder()
        x = torch.rand([10,3,32,32])     
    pred, x_ = model(x)
    print(x.shape,x_.shape,pred.shape)
    assert x_.shape == x.shape

    args["global_model"] = model.encoder
    server = Server(**args)
    clients = {}

    data_loaders, test_loaders = get_client_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, net_dataidx_map)

    criterion_pred = torch.nn.CrossEntropyLoss()
    criterion_rep = torch.nn.MSELoss()

    args["criterion"]={
        "criterion_rep": criterion_rep,
        "criterion_pred": criterion_pred
        }
    
    print("Clusters:")
    task_nodes, node_task = cluster(net_dataidx_map,y_train,args["n_clients"])

    for id in range(args["n_clients"]):
        # dataidxs = net_dataidx_map[id]
        args["id"] = id
        # args["trainloader"], _, _, _ = get_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, dataidxs)
        args["trainloader"] = data_loaders[id]
        args["testloader"] = test_loaders[id]
        args["model"] = copy.deepcopy(model)
        label_map = customize_client_model(args["model"],y_train,net_dataidx_map,id)
        args["label_map"] = label_map

        #Filter data points from global test dataloader to create a cluster test dataloader
        #Only targets in client's label_map will be included in it's cluster testset
        indices = [idx for idx, target in enumerate(test_dl_global.dataset.target) if target in list(label_map.keys())]
        cluster_dataloader = torch.utils.data.DataLoader(Subset(test_dl_global.dataset, indices),batch_size=32,drop_last=True)

        args["cluster_testloader"] = cluster_dataloader
        print("Node",id,"cluster:",len(cluster_dataloader.dataset))

        clients[id] = Client(**args)
    
    for task, nodes in task_nodes.items():
        logger.info('Task %s  :  %s' % (str(task),str(nodes)))


    simulator = MTFLEnv(server=server, clients=clients, communication_rounds=args["comm_round"],n_clients= args["n_clients"],sample_rate=args["sample"],task_nodes=task_nodes, node_task=node_task)

    simulator.run(local_epochs=args["epochs"])


'''
module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl
python eval.py --cf config.yaml
'''
