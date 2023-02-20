from fedlib.utils import Arguments

from fedlib.lib.algo import fedavg

from fedlib.ve import simulator

import fedlib

def main():
    
    args = Arguments()
    args.show_args()
    
    data_loaders, global_test_dl, _ = fedlib.init_dataset(**vars(args.dataset))
    
    model = fedlib.init_model(args.model)
    
    trainer = fedavg(fedlib.get_logger())

    server = fedlib.init_server(n_clients = args.server.n_clients,global_model=model,testloader=global_test_dl,
                                trainer=trainer, communicator=None, sampler=args.server.sampler, device=args.server.device,log_dir=args.server.log_dir)

    clients = fedlib.init_clients(n_clients=args.client.n_clients,model=model,data_loaders=data_loaders,
                            testloader=global_test_dl,lr=args.client.lr,lr_scheduler=args.client.lr_scheduler,
                            criterion=args.client.criterion,optimizer=args.client.optimizer,trainer=trainer, 
                            communicator=None,device=args.client.device,log_dir=args.client.log_dir)
    # print(len(clients))
    runner = simulator(server=server, clients=clients, 
                        communication_rounds=args.general.communication_rounds,
                        n_clients=args.general.n_clients,
                        participate_rate=args.general.participate_rate)
    
    runner.run(**vars(args.trainer))

if __name__ == '__main__':

    main()    