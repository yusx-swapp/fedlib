from typing import Dict
from ..lib.server import Server, ClusterServer
from ..lib.client import Client
from ..utils import get_logger
import random
import functools

class MTFLEnv:
    def __init__(self, server: Server, cluster_servers: ClusterServer, clients: Dict[int, Client], communication_rounds:int, n_clients: int, sample_rate:float,task_nodes=dict, node_task=dict) -> None:
        """_summary_

        Args:
            server (Server): _description_
            clients (List[Client]): _description_
            communication_rounds (int): _description_
        """
        super().__init__()
        self.server = server
        self.cluster_servers = cluster_servers
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
            


    def run(self):

        #Select clients, Train encoder, Train predictors, Test predictors
        for round in range(self.communication_rounds):
            #Phase 0: Sample one clinet per cluster

            selected = []
            for cluster_server in self.cluster_servers:
                selected.append(random.choice(cluster_server._clients))
                    
            #Phase I: Train global encoder on client local data
            
            global_encoder = self.server.get_global_model_params()
            nets_encoders, local_datasize = [], []
            cluster_accuracies, accuracies, losses = [], [], []
            self.logger.info('*******starting rounds %s optimization******' % str(round+1))  
            for id in selected:
                self.logger.info('optimize the %s-th client decoder' % str(id))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                

                #Update client encoder
                client.set_model_params(global_encoder, module_name="encoder")
                client.client_update(layer="encoder")
                
                #Collect client params and datasize
                nets_encoders.append(client.get_model_params(module_name="encoder"))
                local_datasize.append(client.datasize)
            #Update global encoder
            self.server.server_update(nets_encoders=nets_encoders, local_datasize=local_datasize,globa_encoder= global_encoder)

            #Phase II: Train cluster predictors on selected client local data
            global_encoder = self.server.get_global_model_params()

            for idx, cluster_server in enumerate(self.cluster_servers):
                
                #Train global encoder on selected client local data
                cluster_predictor = cluster_server.get_cluster_predictor_params()
                #Pick the selected client in this cluster
                id = selected[idx]
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                
                self.logger.info('optimize TASK-%s client %s-th predictor' % (str(idx),str(id)))

                #Update client encoder and predictor
                client.set_model_params(global_encoder,module_name="encoder")
                client.set_model_params(cluster_predictor,module_name="predictor")
                client.client_update(layer="predictor")

                #Update cluster server's encoder and predictor
                cluster_server.set_cluster_encoder_params(global_encoder)
                client_predictor = client.get_model_params(module_name="predictor")
                cluster_server.set_cluster_predictor_params(client_predictor)
            

            #Phase III: Test predictors of participant client
            for idx, cluster_server in enumerate(self.cluster_servers):
                #Compute cluster server model accuracy on unseen data
                cluster_server_accuracy = cluster_server.eval()["test_accuracy"]
                cluster_accuracies.append(cluster_server_accuracy)
                #Compute sample client accuracy
                id = selected[idx]
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                accuracy = client.eval()["test_accuracy"]
                accuracies.append(accuracy)
                loss = client.eval_decoder()
                losses.append(loss)
                self.logger.info('TASK-{} cluster server accuracy: {:.3f}\t\t Client {}-th local accuracy: {:.3f}\t decoder loss: {:.3f}'.format(idx, cluster_server_accuracy, id, accuracy, loss))

            self.logger.info('Avg cluster server accuracy: {:.3f}'.format(sum(cluster_accuracies)/len(cluster_accuracies)))
            self.logger.info('Avg local accuracy: {:.3f}'.format(sum(accuracies)/len(accuracies)))
            self.logger.info('Avg local decoder loss: {:.3f}'.format(sum(losses)/len(losses)))



            
        
        # for round in range(self.communication_rounds):
        #     selected = self.server.client_sample(n_clients= self.n_clients, sample_rate=self.sample_rate)
        #     globa_encoder = self.server.get_global_model_params()            
            
        #     nets_encoders, local_datasize = [], []
        #     accuracies, losses = [], []
        #     self.logger.info('*******starting rounds %s optimization******' % str(round+1))
            
        #     #Fine tune selected clients
        #     for id in selected:
        #         self.logger.info('optimize the %s-th client' % str(id))
        #         client = self.clients[id]
        #         if id != client.id:
        #             raise IndexError("client id doesn't match")
                
        #         client.set_model_params(globa_encoder, module_name="encoder")
        #         client.client_update()
        #         nets_encoders.append(client.get_model_params(module_name="encoder"))
        #         local_datasize.append(client.datasize)            
            
        #     #Update global encoder
        #     self.server.server_update(nets_encoders=nets_encoders, local_datasize=local_datasize,globa_encoder= globa_encoder)
        #     globa_encoder = self.server.get_global_model_params()
        #     #Update cluster predictors
        #     self.cluster_update(selected)
            
        #     #Collect accuracy from all clients
        #     for id in range(self.n_clients):
        #         #Clients download latest encoder before evaluation
        #         client = self.clients[id]
        #         if id != client.id:
        #             raise IndexError("client id doesn't match")
                
        #         if id in selected:
        #             accuracy = client.eval()["test_accuracy"]
        #             accuracies.append(accuracy)
        #             loss = client.eval_decoder()
        #             losses.append(loss)
        #             self.logger.info('Client {}-th local accuracy: {:.3f}\t decoder loss: {:.3f}'.format(id, accuracy, loss))
                
                

        #     self.logger.info('Global accuracy: {:.3f}'.format(sum(accuracies)/len(accuracies)))
        #     self.logger.info('Global decoder loss: {:.3f}'.format(sum(losses)/len(losses)))
            #Cannot call server.eval() since global model has no predictor/decoder head
            #self.server.eval()
