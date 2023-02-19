from ..utils import get_logger
from ..lib.algo import feddp as trainer
from .simulator import base
__all__ = ['feddp']

class feddp(base):
    trainer = trainer(get_logger())
    print("feddp:",trainer)
    @staticmethod
    def run(local_epochs,pruning_threshold=1e-3, logger=get_logger()):
    
        for round in range(feddp.communication_rounds):
            selected = feddp.server.client_sample(n_clients= feddp.n_clients, sample_rate=feddp.participate_rate)
            
            global_model_param = feddp.server.get_global_model_params()
            nets_params = []
            local_datasize = []
            logger.info('*******starting Rounds %s Optimization******' % str(round+1))
            logger.info('Participate Clients: %s' % str(selected))
            
            for id in selected:
                logger.info('Optimize the %s-th Clients' % str(id))
                client = feddp.clients[id]
                if id != client.id:
                    raise IndexError("id not match")
                
                client.set_model_params(global_model_param)
                client.client_update( epochs=local_epochs,pruning_threshold=pruning_threshold)
                
                nets_params.append(client.get_model_params())
                local_datasize.append(client.datasize)

                metrics = client.eval()
                logger.info(f'*******Client {str(id+1)} Training Finished! Test Accuracy: {str(metrics["test_accuracy"])} ******')


            feddp.server.server_update(nets_params=nets_params, local_datasize=local_datasize,global_model_param= global_model_param)
            metrics = feddp.server.eval()
            logger.info('*******Model Test Accuracy After Server Aggregation: %s *******' % str(metrics["test_accuracy"]))
            logger.info('*******Rounds %s Federated Learning Finished!******' % str(round+1))

