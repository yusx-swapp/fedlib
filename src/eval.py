
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

class NISTAutoencoder(nn.Module):
    def __init__(self):
        super(NISTAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.predictor = nn.Linear(in_features=32, out_features=10, bias=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.predictor(x)
        return x

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
        x = self.predictor(x)
        #x = self.decoder(x)
        return x

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
        x = self.predictor(x)
        #x = self.decoder(x)
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='res18AE', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--pre_epochs', type=int, default=10, help='number of local pre train epochs')
    parser.add_argument('--n_clients', type=int, default=10,  help='number of workers in a distributed cluster')
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
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')

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
    args["test_dataset"] = test_dl_global

    if args["dataset"] in ["mnist","fmnist","femnist"]:
        model = NISTAutoencoder()
        x = torch.rand([10,1,28,28])
        representation = model.encoder(x)
    elif args["dataset"] == "cifar10":
        model = VAE(128,10)
        #model = Cifar10Autoencoder()
        x = torch.rand([10,3,32,32])
        representation, _ = model.encoder(x)
    elif args["dataset"] == "cifar100":
        model = VAE(256,100)
        #model = Cifar100Autoencoder()
        x = torch.rand([10,3,32,32])
        representation, _ = model.encoder(x)
    x_ = model.decoder(representation)
    pred = model(x)
    print(x.shape,x_.shape,pred.shape)

    args["global_model"] = model.encoder
    server = Server(**args)
    clients = {}

    data_loaders = get_client_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, net_dataidx_map)

    criterion_pred = torch.nn.CrossEntropyLoss()
    criterion_rep = torch.nn.MSELoss()

    args["criterion"]={
        "criterion_rep": criterion_rep,
        "criterion_pred": criterion_pred
        }

    for id in range(args["n_clients"]):
        # dataidxs = net_dataidx_map[id]
        args["id"] = id
        # args["trainloader"], _, _, _ = get_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, dataidxs)
        args["trainloader"] = data_loaders[id]
        args["model"] = copy.deepcopy(model)
        clients[id] = Client(**args)

    simulator = MTFLEnv(server=server, clients=clients, communication_rounds=args["comm_round"],n_clients= args["n_clients"],sample_rate=args["sample"])

    simulator.run(local_epochs=args["epochs"])


'''
module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl
python eval.py --cf config.yaml
'''
