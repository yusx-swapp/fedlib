import random
from math import sqrt
import torch.nn.functional as F
import torch
from sklearn.datasets import load_svmlight_file
from torch.autograd.variable import Variable
from torch.utils import data
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
import numpy as np
from .datasets import MNIST_truncated, FashionMNIST_truncated, SVHN_custom, CIFAR10_truncated, Generated, \
    FEMNIST, CelebA_custom, CIFAR100_truncated
from ..utils import mkdirs, get_logger

logger = get_logger()

def download_and_unzip(url, extract_to='.'):
    """Download and unzip a file.

    Args:
        url (string): URL of the zip file.
        extract_to (str, optional): Output directory. Defaults to '.'.
    """
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir):
    # transform = transforms.Compose([transforms.ToTensor()])
    #
    # cifar10_train_ds = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True,
    #                                                  transform=transform)
    # cifar10_test_ds = torchvision.datasets.CIFAR100(datadir, train=False, download=True, transform=transform)
    #
    # X_train = cifar10_train_ds.data
    # y_train = np.array(cifar10_train_ds.targets)
    #
    # X_test = cifar10_test_ds.data
    # y_test = np.array(cifar10_test_ds.targets)
    #
    # # X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    # # X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    #
    # # y_train = y_train.numpy()
    # # y_test = y_test.numpy()

    # return (X_train, y_train, X_test, y_test)

    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)


    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)


    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_celeba_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train = celeba_train_ds.attr[:, gender_index:gender_index + 1].reshape(-1)
    y_test = celeba_test_ds.attr[:, gender_index:gender_index + 1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)


# def load_femnist_data(datadir):
#     transform = transforms.Compose([transforms.ToTensor()])

#     mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=True)
#     mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=True)

#     X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
#     X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

#     X_train = X_train.data.numpy()
#     y_train = y_train.data.numpy()
#     u_train = np.array(u_train)
#     X_test = X_test.data.numpy()
#     y_test = y_test.data.numpy()
#     u_test = np.array(u_test)

#     return (X_train, y_train, u_train, X_test, y_test, u_test)

def load_femnist_data(datadir,split='byclass'):
    # load the FEMNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    femnist_train = datasets.EMNIST(root='./data', split=split, train=True, transform=transform, download=True)
    femnist_test = datasets.EMNIST(root='./data', split=split, train=False, transform=transform, download=True)

    X_train = femnist_train.data.numpy()
    y_train = femnist_train.targets.numpy()
    X_test = femnist_test.data.numpy()
    y_test = femnist_test.targets.numpy()

    
    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:, row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def partition_data(dataset, datadir,  partition, n_parties, beta=0.4,logdir =None,split='byclass',label_node_map=None):
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset == 'femnist':
        #X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
        X_train, y_train, X_test, y_test = load_femnist_data(datadir,split)
    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 == 1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1 > 0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0, 3999, 4000, dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    elif dataset in ('rcv1', 'SUSY', 'covtype'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_train = X_train.todense()
        num_train = int(X_train.shape[0] * 0.75)
        if dataset == 'covtype':
            y_train = y_train - 1
        else:
            y_train = (y_train + 1) / 2
        idxs = np.random.permutation(X_train.shape[0])

        X_test = np.array(X_train[idxs[num_train:]], dtype=np.float32)
        y_test = np.array(y_train[idxs[num_train:]], dtype=np.int32)
        X_train = np.array(X_train[idxs[:num_train]], dtype=np.float32)
        y_train = np.array(y_train[idxs[:num_train]], dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    elif dataset in ('a9a'):
        X_train, y_train = load_svmlight_file("../../../data/{}".format(dataset))
        X_test, y_test = load_svmlight_file("../../../data/{}.t".format(dataset))
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy", X_train)
        np.save("data/generated/X_test.npy", X_test)
        np.save("data/generated/y_train.npy", y_train)
        np.save("data/generated/y_test.npy", y_test)

    n_train = y_train.shape[0]

    if partition == "manual":
        label_idxs = {i:[] for i in np.unique(y_train)}
        if label_node_map == None:
            label_node_map = {i:[] for i in label_idxs.keys()}
            label_node_map[0] = [0,5,10,15,20]
            label_node_map[1] = [0,5,10,15,20]
            label_node_map[2] = [1,6,11,16,21]
            label_node_map[3] = [1,6,11,16,21]
            label_node_map[4] = [2,7,12,17,22]
            label_node_map[5] = [2,7,12,17,22]
            label_node_map[6] = [3,8,13,18,23]
            label_node_map[7] = [3,8,13,18,23]
            label_node_map[8] = [4,9,14,19]
            label_node_map[9] = [4,9,14,19]
        logger.info('MANUAL PARTITION: %s' % str(label_node_map))

        
        for i,label in enumerate(y_train):
            label_idxs[label].append(i)
        
        net_dataidx_map = {i:[] for i in range(n_parties)}
        for label, idxs in label_idxs.items():
            batch_idxs = np.array_split(idxs, len(label_node_map[label]))
            for i, net_id in enumerate(label_node_map[label]):
                net_dataidx_map[net_id] += list(batch_idxs[i])

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            K = 2
            # min_require_size = 100
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'femnist':
            K = 62
            
        N = y_train.shape[0]
        np.random.seed(2020)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        elif dataset == "femnist":
            K = 62
        if num == 10:
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
        else:
            times = [0 for i in range(K)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                        ids += 1

    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # elif partition == "real" and dataset == "femnist":
    #     num_user = u_train.shape[0]
    #     user = np.zeros(num_user + 1, dtype=np.int32)
    #     for i in range(1, num_user + 1):
    #         user[i] = user[i - 1] + u_train[i - 1]
    #     no = np.random.permutation(num_user)
    #     batch_idxs = np.array_split(no, n_parties)
    #     net_dataidx_map = {i: np.zeros(0, dtype=np.int32) for i in range(n_parties)}
    #     for i in range(n_parties):
    #         for j in batch_idxs[i]:
    #             net_dataidx_map[i] = np.append(net_dataidx_map[i], np.arange(user[j], user[j + 1]))

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def get_client_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, n_clients=0, local_test_loader = False,n_worker=32, split="byclass"):
    assert dataset in ('mnist', 'femnist', 'fmnist', 'cifar100','cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY')
    if dataset == 'mnist':
        dl_obj = MNIST_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])

    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])

    elif dataset == 'svhn':
        dl_obj = SVHN_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(0., noise_level, net_id, n_clients)])

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        dl_obj = Generated
        transform_train = None
        transform_test = None

    train_loaders = []
    local_test_loaders = []
    total_train,total_test = 0, 0

    if dataset == "femnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.EMNIST(root='./data', split=split, train=True, transform=transform, download=True)
        test_ds = datasets.EMNIST(root='./data', split=split, train=False, transform=transform, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)     
        #Phuong 09/26 drop_last=False -> True
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
    else:
        train_ds = dl_obj(datadir, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    global_test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    for key, dataid in dataidxs.items():
        

        
        total_train += len(train_ds)
        total_test += len(test_ds)
        if local_test_loader:
            train_dl = data.DataLoader(dataset=torch.utils.data.Subset(train_ds, dataid), batch_size=train_bs, shuffle=True, drop_last=True)     
            #train_dl = data.DataLoader(dataset=torch.utils.data.Subset(train_ds, dataid[:int(0.8*len(dataid))]), batch_size=train_bs, shuffle=True, drop_last=True)     
            #local_test_dl = data.DataLoader(dataset=torch.utils.data.Subset(train_ds, dataid[int(0.8*len(dataid)):]), batch_size=train_bs, shuffle=True, drop_last=True)    
            #local_test_loaders.append(local_test_dl)
            logger.info(f"Client ID:{key+1},\tLocal Train Data Size:{len(train_ds)},\tLocal Test Data Size:{len(test_ds)}")

        else:
            train_dl = data.DataLoader(dataset=torch.utils.data.Subset(train_ds, dataid), batch_size=train_bs, shuffle=True, drop_last=True)   
            #print("Client ID:",key+1, ",\tLocal Train Data Size:",len(train_dl.dataset),len(dataid))  
            logger.info(f"Client ID:{key+1},\tLocal Train Data Size:{len(train_ds)}")
        train_loaders.append(train_dl)

    
    # print("Total train:",total_train,"\t Total test:",total_test)
    
    
    
    return train_loaders, global_test_dl #, local_test_loaders

def get_val_dataloader(dataset, datadir, datasize, val_bs):
    """Load validation data for the proxy similarity computation.

    Args:
        dataset (string): Name of the global dataset.
        datadir (string): Location of the global dataset.
        datasize (int): Number of samples to be drawn from the dataset.
        val_bs (int): Batch size for the data loader.

    Returns:
        torch.utils.data.DataLoader: Data loader for the validation data.
    """
    val_dl = None
    if dataset == 'tinyimagenet':
        if not os.path.exists('./data/tiny-imagenet-200'):
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip','./data/')
        random_ids = np.random.randint(100000, size=datasize)
        val_indices = random_ids

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        val_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(datadir,
                                                transform=transforms.Compose([
                                                transforms.Resize(32), 
                                                transforms.ToTensor(),
                                                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
            #Phuong 09/26 drop_last=False -> True
            batch_size=val_bs, drop_last=True, sampler=SubsetRandomSampler(val_indices))
    
    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated
        transform_val = transforms.Compose([
                transforms.ToTensor(),])
        
        random_ids = np.random.randint(10000, size=datasize)
        val_indices = random_ids

        val_ds = dl_obj(datadir, dataidxs=val_indices, train=True, transform=transform_val, download=True)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=val_bs, shuffle=True, drop_last=False)
    
    elif dataset == "cifar10":
        dl_obj = CIFAR10_truncated
        transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
        random_ids = np.random.randint(10000, size=datasize)
        val_indices = random_ids

        val_ds = dl_obj(datadir, dataidxs=val_indices, train=True, transform=transform_val, download=True)
        val_dl = torch.utils.data.DataLoader(dataset=val_ds, batch_size=val_bs, shuffle=True, drop_last=False)


    return val_dl

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0, n_worker=32, split="byclass"):
    if dataset in ('mnist', 'femnist', 'fmnist', 'cifar100','cifar10', 'svhn', 'generated', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        # elif dataset == 'femnist':
        #     # dl_obj = FEMNIST

        #     # Additional transformation from grayscale -> RGB needed here
        #     # transform_train = transforms.Compose([
        #     #     transforms.ToTensor(),
        #     #     AddGaussianNoise(0., noise_level, net_id, total),
        #     #     transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        #     # transform_test = transforms.Compose([
        #     #     transforms.ToTensor(),
        #     #     AddGaussianNoise(0., noise_level, net_id, total),
        #     #     transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
        #     dl_obj = FEMNIST
        #     transform_train = transforms.Compose([
        #         transforms.ToTensor(),
        #         AddGaussianNoise(0., noise_level, net_id, total)])
        #     transform_test = transforms.Compose([
        #         transforms.ToTensor(),
        #         AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            # cifar_tran_train = [
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ]
            # cifar_tran_test = [
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ]
            # transform_train = transforms.Compose(cifar_tran_train)
            # transform_test = transforms.Compose(cifar_tran_test)
            # train_ds = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True,
            #                                          transform=transform_train)
            # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_bs, shuffle=True,
            #                                            num_workers=n_worker, pin_memory=True, sampler=None)
            # test_ds = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True,
            #                                         transform=transform_test)
            # test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_bs, shuffle=False,
            #                                          num_workers=n_worker, pin_memory=True)
            # return train_dl, test_dl, train_ds,
        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None

        if dataset == "femnist":
            transform = transforms.Compose([transforms.ToTensor()])

            train_ds = datasets.EMNIST(root='./data', split=split, train=True, transform=transform, download=True)
            test_ds = datasets.EMNIST(root='./data', split=split, train=False, transform=transform, download=True)
            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)     
            #Phuong 09/26 drop_last=False -> True
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)
            return train_dl, test_dl, train_ds, test_ds

        if dataidxs is not None:
            train_ds = dl_obj(datadir, dataidxs=dataidxs[:int(len(dataidxs)*0.8)], train=True, transform=transform_train, download=True)
            test_ds = dl_obj(datadir, train=True,dataidxs=dataidxs[int(len(dataidxs)*0.8):], transform=transform_test, download=True)
        else:
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
            test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        #Phuong 09/26 drop_last=False -> True
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)     
        #Phuong 09/26 drop_last=False -> True
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl, train_ds, test_ds


def sample_dataloader(dataset, datadir, datasize, train_bs, test_bs=32):

    train_dl = None

    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        random_ids = np.random.randint(50000, size=datasize)
        print(random_ids)
        train_ds = dl_obj(datadir, dataidxs=random_ids, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        #Phuong 09/26 drop_last=False -> True
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        #Phuong 09/26 drop_last=False -> True
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)


    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        random_ids = np.random.randint(50000, size=datasize)
        print(random_ids)
        train_ds = dl_obj(datadir, dataidxs=random_ids, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        #Phuong 09/26 drop_last=False -> True
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        #Phuong 09/26 drop_last=False -> True
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    elif dataset == 'mnist':
        dl_obj = MNIST_truncated

        transform_train = transforms.Compose([
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.ToTensor()])
        random_ids = np.random.randint(50000, size=datasize)
        print(random_ids)
        train_ds = dl_obj(datadir, dataidxs=random_ids, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        #Phuong 09/26 drop_last=False -> True
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
        #Phuong 09/26 drop_last=False -> True
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    elif dataset =='femnist':
        dl_obj = FEMNIST
        transform_train = transforms.Compose([
            transforms.ToTensor()])
        transform_test = transforms.Compose([
            transforms.ToTensor()])
        random_ids = np.random.randint(50000, size=datasize)
        print(random_ids)
        train_ds = dl_obj(datadir, dataidxs=random_ids, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
        #Phuong 09/26 drop_last=False -> True
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)


    elif dataset == 'cinic10':

        random_ids = np.random.randint(70000, size=datasize)

        train_indices = random_ids

        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        
        train_dl = torch.utils.data.DataLoader(
            # Phuong: change to new CINIC10 without Cifar10
            datasets.ImageFolder(datadir,
                                             transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Lambda(lambda x: F.pad(
                                                            Variable(x.unsqueeze(0), requires_grad=False),
                                                            (4, 4, 4, 4), mode='reflect').data.squeeze()),
                                                        transforms.ToPILImage(),
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
            #Phuong 09/26 drop_last=False -> True
            batch_size=train_bs, drop_last=True, sampler=SubsetRandomSampler(train_indices))


    elif dataset == 'tinyimagenet':
        
        random_ids = np.random.randint(100000, size=datasize)

        train_indices = random_ids

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        train_dl = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(datadir,
                                             transform=transforms.Compose([
                                                transforms.Resize(32), 
                                                transforms.ToTensor(),
                                                # Phuong 09/26 change (mean, std) -> same as pretrained imagenet
                                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])),
            #Phuong 09/26 drop_last=False -> True
            batch_size=train_bs, drop_last=True, sampler=SubsetRandomSampler(train_indices))

   


    return train_dl
