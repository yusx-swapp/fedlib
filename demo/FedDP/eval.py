from fedlib.utils import Arguments
from fedlib.ve import FEDDFEnv

from fedlib.ve import feddp



def main():
    
    args = Arguments()
    args.show_args()
    
    data_loaders, global_test_dl, _ = feddp.init_dataset(args.dataset)
    
    model = feddp.init_model(args.model)
    
    server = feddp.init_server(args.server,global_model=model,test_dataset=global_test_dl,
                        communicator=None)
    
    clients = feddp.init_clients(args.client,model=model,data_loaders=data_loaders,
                            testloader=global_test_dl,
                            communicator=None)
    
    simulator = FEDDFEnv(server=server, clients=clients, 
                        communication_rounds=args.general.communication_rounds,
                        n_clients=args.general.n_clients,
                        participate_rate=args.general.participate_rate)
    
    simulator.run(local_epochs=args.optimization.local_epochs,pruning_threshold=args.algo.pruning_threshold)

if __name__ == '__main__':

    main()    