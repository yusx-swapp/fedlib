from msilib.schema import Class
from tkinter import NO

from fedlib.src.fedlib import runner


class runner():
    def __init__(self, config) -> None:
        args = load_arguments(config)




    def run(config=None,):



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