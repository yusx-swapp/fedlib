import torch
from torch import nn

from ....utils import get_logger
from ..base import BaseTrainer
logger = get_logger()

class Trainer(BaseTrainer):

    def train(self, model:nn.Module, dataloader , criterion, optimizer, epochs:int, device):
            
        avg_acc = 0.0
        avg_kacc = 0.0
        k_nets = []
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]

            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            n_epoch = args.epochs



            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            net.to(device)

            kd_agent = Distiller(lr=lr,epochs=n_epoch,device=device)

            nets_cohort = []
            optimizers = []

            nets_cohort.append(net)
            net_optimizer = optim.SGD(net.parameters(), lr)
            optimizers.append(net_optimizer)

            l_g_k = copy.deepcopy(g_k)
            k_nets.append(l_g_k)
            nets_cohort.append(l_g_k)
            gk_optimizer = optim.SGD(l_g_k.parameters(), lr)
            optimizers.append(gk_optimizer)



            #### if use the local test_dl (also need modify get_dataloader function):
            '''
            nets_cohort, lr_ = kd_agent.mutual_kd(nets_cohort,train_dl_local, test_dl_local,optimizers,s_save_path=None)
            acc =compute_accuracy(nets_cohort[0],test_dl_local,device=device)
            print("net %d final test acc %f" % (net_id, acc))
            logger.info("net %d final test acc %f" % (net_id, acc))
            acc_k = compute_accuracy(nets_cohort[1],test_dl_local,device=device)
            print("net %d knowledge network test acc %f" % (net_id, acc_k))
            logger.info("net %d knowledge network test acc %f" % (net_id, acc_k))
            '''


            #### if use the global test_dl:

            nets_cohort, lr_ = kd_agent.mutual_kd(nets_cohort, train_dl_local, test_dl_global, optimizers, s_save_path=None)

            acc = compute_accuracy(nets_cohort[0], test_dl_global, device=device)
            print("net %d final test acc %f" % (net_id, acc))
            logger.info("net %d final test acc %f" % (net_id, acc))
            acc_k = compute_accuracy(nets_cohort[1], test_dl_global, device=device)
            print("net %d knowledge network test acc %f" % (net_id, acc_k))
            logger.info("net %d knowledge network test acc %f" % (net_id, acc_k))


            avg_acc += acc
            avg_kacc += acc_k

        avg_acc /= len(selected)
        avg_kacc /= len(selected)
        logger.info("avg test acc after local update %f" % avg_acc)
        logger.info("avg knowledge test acc after local update %f" % avg_kacc)

        nets_list = list(nets.values())
        return nets_list,k_nets,lr_

        def aggregate(self, **kwargs):        
                """fedavg aggregation
                kwargs:
                    nets_params: 
                    local_datasize:
                    global_para: 

                Returns:
                    global_para: _description_
                """
                nets_params = kwargs["nets_params"]
                local_datasize = kwargs["local_datasize"]
                global_model_param = kwargs["global_model_param"]

                total_data_points = sum(local_datasize)
                fed_avg_freqs = [size/ total_data_points for size in local_datasize]
                
                
                for idx, net_para in enumerate(nets_params):
                    if idx == 0:
                        for key in net_para:
                            global_model_param[key] = net_para[key] * fed_avg_freqs[idx]
                    else:
                        for key in net_para:
                            global_model_param[key] += net_para[key] * fed_avg_freqs[idx]

                return global_model_param

    def test(self, model, test_data, device):
        

        model.to(device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }

        """
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        """
        # if args.dataset == "stackoverflow_lr":
        #     criterion = nn.BCELoss(reduction="sum").to(device)
        # else:
            # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                # if args.dataset == "stackoverflow_lr":
                #     predicted = (pred > 0.5).int()
                #     correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                #     true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                #     precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                #     recall = true_positive / (target.sum(axis=-1) + 1e-13)
                #     metrics["test_precision"] += precision.sum().item()
                #     metrics["test_recall"] += recall.sum().item()
                # else:
                #     _, predicted = torch.max(pred, 1)
                #     correct = predicted.eq(target).sum()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:

        return False