import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
from typing import Dict
from ..lib.server import Server
from ..lib.client import Client
from ..utils import get_logger

class CSFLEnv:
    def __init__(self, server: Server, clients: Dict[int, Client], communication_rounds:int, n_clients: int, sample_rate:float, val_dl_global) -> None:
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
        
        self.n_clients = n_clients
        self.sample_rate = sample_rate
        self.logger = get_logger()

        self.val_dl_global = val_dl_global
        #Pretrain selected nodes to compute local gradients for 10 epochs
        #local_pre_training(nets, selected, args, net_dataidx_map, args.pre_epochs, test_dl = test_dl_global, device=device)


    #Todo initialize by configuration
    def init_config(self) -> None:
        pass

    # TODO initilize server
    def init_server(self) -> None:
        pass
    
    # TODO initilize clients
    def init_clients(self) -> None:
        pass

    def pretrain(self,local_epochs) -> None:
        """Pretrain all clients for a limited number of epochs
        and compute the clients output on the global val dataset.
        Done on client side.
        """

        nets_encoders, local_datasize = [], []
        accuracies, losses = [], []
        self.logger.info('******* start PRE-TRAINING ******')
        for id in range(self.n_clients):
            self.logger.info('Optimize client %s' % str(id))
            client = self.clients[id]
            if id != client.id:
                raise IndexError("client id doesn't match")
            
            client.client_update( epochs=local_epochs)
            client.test_on_global()
            self.logger.info('Client {} finished pre-training'.format(id))
        self.logger.info('******* end PRE-TRAINING ******')
    
    def cluster(self,sim_measure,num_clusters) -> None:
        """Compute the similarity matrix among clients
        and cluster them using KMeans + GCN model. Done
        on server side.
        """
        
        kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)
        for i in range(self.n_clients):
            client_i = self.clients[i]
            if i != client_i.id:
                raise IndexError("client id doesn't match")
            for j in range(self.n_clients):
                client_j = self.clients[j]
                if j != client_j.id:
                    raise IndexError("client id doesn't match")
                if sim_measure == "kl":
                    self.server._sim_matrix[i][j] = kl_loss(F.log_softmax(client_i._global_output,dim=1),F.log_softmax(client_j._global_output,dim=1))


        if any(None in sublist for sublist in self.server._sim_matrix):
            raise IndexError("Can't cluster clients before the SIM_MAT is complete.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.server._sim_matrix)
        assert len(kmeans.labels_) == self.n_clients
        self.server._clusters = {i:[] for i in range(num_clusters)}
        #Map cluster_id : [client_id,..]
        for idx,label in enumerate(kmeans.labels_):
            self.server._clusters[label].append(idx)

    def run(self,local_epochs):
        
        for round in range(self.communication_rounds):
            selected = self.server.client_sample(n_clients= self.n_clients, clusters=self.server._clusters , sample_rate=self.sample_rate)
            globa_model = self.server.get_global_model_params()            
            
            nets_encoders, local_datasize = [], []
            self.logger.info('*******starting rounds %s optimization******' % str(round+1))
            self.logger.info('CLUSTERS: %s' % str(self.server._clusters))
            self.logger.info('SELECTED %d: %s' % (len(selected), str(selected)))
            
            #Fine tune selected clients
            for id in selected:
                self.logger.info('optimize the %s-th clients' % str(id))
                client = self.clients[id]
                if id != client.id:
                    raise IndexError("client id doesn't match")
                
                client.client_update( epochs=local_epochs)
                nets_encoders.append(client.get_model_params())
                local_datasize.append(client.datasize)            
            
            self.server.server_update(nets_encoders=nets_encoders, local_datasize=local_datasize,globa_model= globa_model)
            
            #globa_model = self.server.get_global_model_params()
            
            #Collect accuracy from all clients
            # for id in range(self.n_clients):
            #     #Clients download latest encoder before evaluation
            #     client = self.clients[id]
            #     if id != client.id:
            #         raise IndexError("client id doesn't match")
            #     client.set_model_params(globa_model)
            #     accuracy = client.eval()["test_accuracy"]
            #     accuracies.append(accuracy)
            #     if id in selected:
            #         loss = client.eval_decoder()
            #         self.logger.info('Client {}-th local accuracy: {:.3f}\t decoder loss: {:.3f}'.format(id, accuracy, loss))
            #         losses.append(loss)
            #     else:
            #         self.logger.info('Client {}-th local accuracy: {:.3f}\t decoder loss: {}'.format(id, accuracy, "-"))

            accuracy = self.server.eval()["test_accuracy"]                

            self.logger.info('Global accuracy: {:.3f}'.format(accuracy))
            #Cannot call server.eval() since global model has no predictor/decoder head
            #self.server.eval()
