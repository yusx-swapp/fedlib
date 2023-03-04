from typing import Dict
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger

class MTFLEnv:
    def __init__(self, server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, sample_rate:float,task_nodes=dict, node_task=dict) -> None:
        """_summary_

        Args:
            server (Server): _description_
            clients (List[Client]): _description_
            communication_rounds (int): _description_
        """
        super().__init__()
        self.server = server
        self.clients = clients
        self.communication_rounds = communication_rounds
        self.task_nodes = task_nodes
        self.node_task = node_task
        
        self.n_clients = n_clients
        self.sample_rate = sample_rate
        self.logger = get_logger()


    #Todo initialize by configuration
    def init_config(self) -> None:
        pass

    # TODO initilize server
    def init_server(self) -> None:
        pass
    
    # TODO initilize clients
    def init_clients(self) -> None:
        pass

    def aggregate_predictors(self, nets_predictors,local_datasize, client_predictor ):        
        """fedavg aggregation
        kwargs:
            nets_encoders: 
            local_datasize:
            globa_encoder: 

        Returns:
            globa_encoder: _description_
        """
        # nets_params = kwargs["nets_params"]
        # local_datasize = kwargs["local_datasize"]
        # global_model_param = kwargs["global_model_param"]

        total_data_points = sum(local_datasize)
        fed_avg_freqs = [size/ total_data_points for size in local_datasize]
        
        
        for idx, net_para in enumerate(nets_predictors):
            if idx == 0:
                for key in net_para:
                    client_predictor[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    client_predictor[key] += net_para[key] * fed_avg_freqs[idx]

        return client_predictor
    
    def cluster_update(self,selected):
        """Aggregate cluster predictors if any node in the cluster
        was selected for local fine-tuning in the current round.

        Args:
            selected (list): List of selected clients trained in this round.
        """
        for task,nodes in self.task_nodes.items():
            net_predictors, local_datasize = [], []
            #Count the number of nodes in cluster that will have preds aggregated
            aggregate = 0
            for id in nodes:
                if id in selected:
                    aggregate += 1
                    net_predictors.append(self.clients[id].get_model_params(module_name="predictor"))
                    local_datasize.append(self.clients[id].datasize)
            #If params to aggregate, aggregate cluster
            if aggregate > 0:
                for id in nodes:
                    client_predictor = self.clients[id].get_model_params(module_name="predictor")
                    predictor_parameters = self.aggregate_predictors(net_predictors, local_datasize,client_predictor)
                    self.clients[id]._model.predictor.load_state_dict(predictor_parameters)
            


    def run(self,local_epochs):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.sample_rate)
            globa_encoder = self.server.get_global_model_params()            
            
            nets_encoders, local_datasize = [], []
            accuracies, losses = [], []
            self.logger.info('*******starting rounds %s optimization******' % str(round+1))
            
            #Fine tune selected clients
            for id in selected:
                self.logger.info('optimize the %s-th client' % str(id))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                
                client.set_model_params(globa_encoder, module_name="encoder")
                client.client_update( epochs=local_epochs)
                nets_encoders.append(client.get_model_params(module_name="encoder"))
                local_datasize.append(client.datasize)            
            
            #Update global encoder
            self.server.server_update(nets_encoders=nets_encoders, local_datasize=local_datasize,globa_encoder= globa_encoder)
            globa_encoder = self.server.get_global_model_params()
            #Update cluster predictors
            self.cluster_update(selected)
            
            #Collect accuracy from all clients
            for id in range(self.n_clients):
                #Clients download latest encoder before evaluation
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                
                if id in selected:
                    accuracy = client.eval()["test_accuracy"]
                    accuracies.append(accuracy)
                    loss = client.eval_decoder()
                    losses.append(loss)
                    self.logger.info('Client {}-th local accuracy: {:.3f}\t decoder loss: {:.3f}'.format(id, accuracy, loss))
                
                

            self.logger.info('Global accuracy: {:.3f}'.format(sum(accuracies)/len(accuracies)))
            self.logger.info('Global decoder loss: {:.3f}'.format(sum(losses)/len(losses)))
            #Cannot call server.eval() since global model has no predictor/decoder head
            #self.server.eval()
