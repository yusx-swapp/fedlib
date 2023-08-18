import torch
import datetime
import math
import torch
from torchvision import models as models
import numpy as np
import torchvision
from efficientnet_pytorch import EfficientNet
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy
import pickle
import argparse
from fedlib.utils import get_logger, init_logs
from fedlib.ve.csfl import CSFLEnv
from fedlib.lib import Server, Client
from fedlib.networks import resnet20, NeuralNet, vgg11
from fedlib.lib.sampler import stratified_cluster_sampler
from fedlib.lib.algo.fedcs import Trainer
from fedlib.datasets import partition_data, get_dataloader,get_client_dataloader, get_val_dataloader
from fedlib.networks import VAE, NISTAutoencoder, Cifar100Autoencoder, Cifar10Autoencoder

from torch import nn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='res20', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
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
    parser.add_argument('--split', type=str, default='byclass', help='How to split FEMNIST dataset')
    parser.add_argument('--lr_scheduler', type=str, default='ExponentialLR', help='Learning rate scheduler')
    parser.add_argument('--decay_rate', type=float, default=.99, help='Learning rate scheduler decay rate')
    parser.add_argument('--num_clusters', type=int, default=0, help='Number of clusters: default=0 meaning log(n)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    
    NOW = str(datetime.datetime.now()).replace(" ","--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    # log_dir = './logs/{}_dataset[{}]_model[{}]_partition[{}]_nodes[{}]_rounds[{}]_frac[{}]_local_ep[{}]_local_bs[{}]_beta[{}]_noise[{}]/'. \
    #     format(NOW,args.dataset, args.model, args.partition, args.n_clients, args.comm_round, args.sample,args.epochs, args.batch_size, args.beta, args.noise)
    log_dir = './proofs/{}_dataset[{}]_model[{}]_partition[{}]_algo[{}]_clients[{}]_rounds[{}]_frac[{}]_lr[{}]_scheduler[{}]_decay_rate[{}]_local_bs[{}]_beta[{}]_noise[{}]/'. \
        format(NOW,args.dataset, args.model, args.partition, "CSFL", args.n_clients, args.comm_round, args.sample, args.lr, args.lr_scheduler, args.decay_rate, args.batch_size, args.beta, args.noise)
    
    args = vars(args)
    
    init_logs(log_file_name, args, log_dir)
    logger = get_logger()

    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['sample_fn'] = stratified_cluster_sampler
    args['trainer'] = Trainer(logger)
    args['communicator'] = None
    args["datadir"] = "./data"
    args["log_dir"] = log_dir
    args["lr_scheduler"] = "ExponentialLR" 
    args["sampler"] = "stratified_cluster_sampler"
    args['num_clusters'] = math.ceil(math.log2(args['n_clients'])) if args['num_clusters'] == 0 else args['num_clusters']
    args["sim_measure"] = "kl" #optionally EMD for earth movers distance

    print("Args:",args)
    # label_node_map[0] = [0,5,10,15,20]
    # label_node_map[1] = [0,5,10,15,20]
    # label_node_map[2] = [1,6,11,16,21]
    # label_node_map[3] = [1,6,11,16,21]
    # label_node_map[4] = [2,7,12,17,22]
    # label_node_map[5] = [2,7,12,17,22]
    # label_node_map[6] = [3,8,13,18,23]
    # label_node_map[7] = [3,8,13,18,23]
    # label_node_map[8] = [4,9,14,19]
    # label_node_map[9] = [4,9,14,19]
    label_node_map = None
    # label_node_map = {}
    # label_node_map[0] = [0,1,2,3,4]
    # label_node_map[1] = [0,1,2,3,4]
    # label_node_map[2] = [0,1,2,3,4,5,6]
    # label_node_map[3] = [5,6,7,8,9]
    # label_node_map[4] = [5,6,7,8,9,10,11]
    # label_node_map[5] = [10,11,12,13]
    # label_node_map[6] = [10,11,12,13,14]
    # label_node_map[7] = [13,14,15,16]
    # label_node_map[8] = [13,14,15,16,17,18]
    # label_node_map[9] = [19,20,21,22,23]

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    args["dataset"],args["datadir"], args['partition'], args['n_clients'], beta=args['beta'],split=args['split'],label_node_map=label_node_map)
    n_classes = len(np.unique(y_train))
    print("X_train.shape:",X_train.shape,"\ty_train.shape:",y_train.shape)
    print("X_test.shape:",X_test.shape,"\ty_test.shape:",y_test.shape)
    print("LABELS:",n_classes)
    args["n_classes"] = n_classes
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args["dataset"],
                                                                                    args["datadir"],
                                                                                      args["batch_size"],
                                                                                      32,
                                                                                      split=args['split'])
    print(args)
    #Setup global dataset dataloader
    if args["dataset"] in ["cifar10","cifar100"]:
        val_dl_global = get_val_dataloader("tinyimagenet", "./data/tiny-imagenet-200/", 1000, 32)
    elif args["dataset"] in ["mnist","femnist"]:
        val_dl_global = get_val_dataloader("fmnist", "./data/", 1000, 32)
    elif args["dataset"] in ["tinyimagenet"]:
         val_dl_global = get_val_dataloader("cifar10", "./data/", 1000, 32)
    
    args["val_dl_global"] = val_dl_global

    print("train_dl_global:",len(train_dl_global.dataset))
    print("test_dl_global:",len(test_dl_global.dataset))
    
    args["test_dl_global"] = test_dl_global

    if args["dataset"] in ["mnist","fmnist"]:
        #Custom NIST FFNN model
        input_size, hidden_size, num_classes = 784, 10, 10
        model = NeuralNet(input_size, hidden_size, num_classes)
    elif args["dataset"] in ["femnist"]:
        #Custom FENIST FFNN model
        input_size, hidden_size, num_classes = 784, 128, 62
        model = NeuralNet(input_size, hidden_size, num_classes)
    elif args["dataset"] == "cifar10":
        #Use custom resnet for cifar10
        if args["model"] == "res20":
            #model = resnet20(10)
            model = models.resnet18(pretrained=False)
        elif args["model"] == "vgg11":
            model = vgg11(10)
        elif args["model"] == "mobilenetv3":
            model = torchvision.models.mobilenet_v3_small(weights=None, num_classes=n_classes)
            #net = MobileNetV3('small',n_classes)
        elif args["model"] == "efficientnet":
            model = EfficientNet.from_name('efficientnet-b0', num_classes=n_classes)
    elif args["dataset"] == "cifar100":
        #Use torchvision resnet for cifar100
        #model = models.resnet18(num_classes=100)
        if args["model"] == "res20":
            model = resnet20(100)
        elif args["model"] == "vgg11":
            model = vgg11(100)
   

    args["global_model"] = model
    server = Server(**args)
    clients = {}

    data_loaders, test_loaders = get_client_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, net_dataidx_map,split=args['split'])

    criterion_pred = torch.nn.CrossEntropyLoss()

    args["criterion"]= criterion_pred
    
    for id in range(args["n_clients"]):
        # dataidxs = net_dataidx_map[id]
        args["id"] = id
        # args["trainloader"], _, _, _ = get_dataloader(args["dataset"], args["datadir"], args['batch_size'], 32, dataidxs)
        args["trainloader"] = data_loaders[id]
        args["model"] = copy.deepcopy(model)
        print("Client:",id)
        clients[id] = Client(**args)

    simulator = CSFLEnv(server=server, clients=clients, communication_rounds=args["comm_round"],
                        n_clients= args["n_clients"],sample_rate=args["sample"], val_dl_global=val_dl_global)

    simulator.pretrain(local_epochs=args["epochs"])
    print("Done pretraining")
    simulator.cluster(sim_measure=args["sim_measure"],num_clusters=args["num_clusters"])
    print("Done clustering")
    #simulator.run(local_epochs=args["epochs"])

    #Compute similarities
    #Measure sample entropies
    rand_entropies, strat_entropies = [], []
    for i in range(500):
        random_sampler_entropy = simulator.compare_sample_entropies(sampler='random')
        rand_entropies.append(round(random_sampler_entropy,2))
    
    for i in range(500):
        clustered_sampler_entropy = simulator.compare_sample_entropies(sampler='stratified')
        strat_entropies.append(round(clustered_sampler_entropy,2))
    
    logger.info(f'random entropies: {rand_entropies}')
    logger.info(f'strat entropies: {strat_entropies}')
    print("Done with computing entropy")

    mechanism = 'stratified'
    cluster_sims, avg = simulator.compute_cluster_activation_similarities(mechanism=mechanism)
    logger.info(f"{mechanism} mean distance : {round(avg,2)}")
    for k,cluster_sim in enumerate(cluster_sims):
        logger.info(f"\t -> cluster-{k} : {cluster_sim}")
    
    mechanism = 'random'
    cluster_sims, avg = simulator.compute_cluster_activation_similarities(mechanism=mechanism)
    logger.info(f"{mechanism} HLF mean distance : {round(avg,2)}")
    for k,cluster_sim in enumerate(cluster_sims):
        logger.info(f"\t -> cluster-{k} : {cluster_sim}")
        
    print("Done with computing HLF similarities")

    # transform = ToTensor()
    # cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # dataloader = DataLoader(cifar_test, 
    #                         batch_size=32, # according to your device memory
    #                         shuffle=False)  # Don't forget to seed your dataloader

    subset_dataset = Subset(test_dl_global.dataset, indices=range(1000))
    subset_datalaoder = DataLoader(subset_dataset,batch_size=32)

    intra_results, inter_results = simulator.compute_cka(device='cuda',dataloader=subset_datalaoder)

    plt.imshow(intra_results['CKA'])
    plt.xlabel('Layers model_1a')
    plt.ylabel('Layers model_1b')
    plt.colorbar()
    plt.savefig(log_dir + 'Intra-cluster.png')
    plt.close()

    plt.imshow(inter_results['CKA'])
    plt.xlabel('Layers model_1a')
    plt.ylabel('Layers model_2')
    plt.colorbar()
    plt.savefig(log_dir + 'Inter-cluster.png')
    plt.close()

    with open(log_dir + "intra_cka_results.pickle", "wb") as file:
        pickle.dump(intra_results, file)
    
    with open(log_dir + "inter_cka_results.pickle", "wb") as file:
        pickle.dump(inter_results, file)
    


'''
module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl
python eval.py --cf config.yaml
'''
