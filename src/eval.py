from RaFL.utils import compute_acc
import torch

import datetime

import numpy as np

from RaFL.lib.fl import local_update, cloud_update

from RaFL.lib.fl import init_fl
from RaFL.utils import init_logs
from RaFL.utils import load_arguments
from RaFL.utils import partition_data, get_dataloader, sample_dataloader
from RaFL.utils import save_checkpoint

if __name__ == '__main__':

    args = load_arguments()

    ########################################create log file###########################################################
    if args.log_file_name is None:
        log_file_name = args.supernet_name + '_' + str(args.n_parties) + '_sample:' + str(
            args.sample) + '-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        log_file_name = args.log_file_name

    logger = init_logs(log_file_name, args=args, log_dir=args.log_dir)



    ###################################################################################################
    device = torch.device(args.device)
    logger.info(device)
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")

    '''
    prepare data
    '''
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.log_dir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    logger.info("Initializing nets")
    
    # generate resource constraints
    resource_constraints = list(np.random.randint(low=30, high = 100, size=args.n_parties))
    
    logger.info("Resource constraints for edge devices: "+str(resource_constraints))

    nets, glob_knwl_net, nets_MACs = init_fl(
        resource_constraints, 
        args.supernet_name, 
        args.tolerance,
        args.max_try,
        args.image_size,
        args.num_classes
    )

    logger.info("Network Macs: "+str(nets_MACs))

   
    global_para = glob_knwl_net.state_dict()

    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round) + "#" * 100)

        # select clients
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]

        # local updates:
        _, k_nets = local_update(nets, glob_knwl_net, selected, args, net_dataidx_map, test_dl_global, logger,
                                     lr=args.lr, device=device)

        # cloud updates
        train_dl_global = sample_dataloader(args.cloud_dataset,
                                               args.cloud_datadir,
                                               args.cloud_datasize,
                                               args.cloud_batch_size)
        cloud_update(k_nets, glob_knwl_net, train_dl_global, args.lr_g, args.cloud_epochs, device, args.kd_weight)

        acc_g, _ = compute_acc(val_loader=test_dl_global, device=device, model = glob_knwl_net)
        print("communication round %d global model test acc %f" % (round, acc_g))
        logger.info("communication round %d global model test acc %f" % (round, acc_g))

    save_checkpoint({
        'state_dict': glob_knwl_net.module.state_dict() if isinstance(glob_knwl_net,
                                                                     torch.nn.DataParallel) else glob_knwl_net.state_dict(),
        # 'acc': test_acc,

    }, checkpoint_dir=args.ckp_dir)



'''
module load miniconda3/4.3.30-qdauveb
source activate /work/LAS/jannesar-lab/sixing/.conda/envs/rafl
python eval.py --cf config.yaml
'''
