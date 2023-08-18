import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from sklearn.cluster import KMeans
from torch_cka import CKA
from typing import Dict
import random
from fedlib.lib.sampler import *
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
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(torch.tensor(self.server._sim_matrix).cpu().data.numpy())
        assert len(kmeans.labels_) == self.n_clients
        self.server._clusters = {i:[] for i in range(num_clusters)}
        #Map cluster_id : [client_id,..]
        for idx,label in enumerate(kmeans.labels_):
            self.server._clusters[label].append(idx)
    
    def compute_relative_entropy(self,p, q):
        entropy = 0
        for i in range(len(p)):
            if p[i] != 0 and q[i] != 0:
                entropy += p[i] * math.log(p[i] / q[i])
        return entropy

    def sum_class_probs(self,client_probs,n_classes):
        sum_prob = {}
        for label in range(n_classes):
            sum_prob[label] = sum([client_prob[label] for client_prob in client_probs]) / len(client_probs)
        return sum_prob
    
    def compare_sample_entropies(self,sampler='random'):
        if sampler == 'random':
            samples = random_sampler(self.n_clients, self.sample_rate)
        elif sampler == 'stratified':
            samples = stratified_cluster_sampler(self.n_clients, self.server._clusters, self.sample_rate)
        else:
            raise Exception('Unknown sampler specified!')
        server_class_probs = self.server.class_probs()
        client_probs = []
        for id in samples:
            client = self.clients[id]
            if id != client.id:
                raise IndexError("client id doesn't match")
            client_class_probs = client.class_probs()
            client_probs.append(client_class_probs)
        print(f'{sampler} --> {client_probs}')
        summed_client_prob = self.sum_class_probs(client_probs,self.server._n_classes)
        #self.logger.info(f"{sampler} summed_client_prob : {summed_client_prob}")
        entropy = self.compute_relative_entropy(summed_client_prob,server_class_probs)
        self.logger.info(f"{sampler} sample entropy : {round(entropy,2)}")
        return entropy
    
    def vector_similarity(self,vecs1,vecs2):
        if len(vecs1) != len(vecs2):
            raise ValueError("Vector size mismatch")
        total_distance = 0
        for v1,v2 in zip(vecs1,vecs2):
            total_distance += torch.dist(v1, v2).item()
        return total_distance / len(vecs1)    

    def average_high_level_feature_similarity(self,client_ids):
        client_vectors = {id:[] for id in client_ids}
        client_dist = {id:0 for id in client_ids}
        for id in client_ids:
            client = self.clients[id]
            if id != client.id:
                raise IndexError("client id doesn't match")
            client_vectors[client.id] = client.get_latent_vectors()
        
        #Compute pairwise client HLF similarity matrix
        for id1 in client_ids:
            for id2 in client_ids:
                client_dist[id1] += self.vector_similarity(client_vectors[id1],client_vectors[id2])
            client_dist[id1] /= len(client_ids)

        return client_dist

    def split_list_into_m_sublists(self,lst, m):
        sublist_size = len(lst) // m
        remainder = len(lst) % m

        sublists = []
        start = 0

        for i in range(m):
            sublist_length = sublist_size + (1 if i < remainder else 0)
            sublist = lst[start:start+sublist_length]
            sublists.append(sublist)
            start += sublist_length

        return sublists

    def compute_cluster_activation_similarities(self,mechanism='stratified'):
        cluster_sims = []
        total = 0
        if mechanism == 'stratified':
            for k,cluster in self.server._clusters.items():
                cluster_sims.append(self.average_high_level_feature_similarity(cluster))
        elif mechanism == 'random':
            clients = list(self.clients.keys()).copy()
            random.shuffle(clients)
            clusters = self.split_list_into_m_sublists(clients,len(self.server._clusters))
            rand_clusters = {i:cluster  for i,cluster in enumerate(clusters)}
            for k,cluster in rand_clusters.items():
                cluster_sims.append(self.average_high_level_feature_similarity(cluster))

        for cluster_sim in cluster_sims:
            total += sum(cluster_sim.values())
        avg = total / self.n_clients
        return cluster_sims, avg

    def compute_cka(self,device,dataloader):
        #Select 2 random clusters
        idx_1,idx_2 = random.sample(self.server._clusters.keys(), 2)
        cluster_1,cluster_2 = self.server._clusters[idx_1], self.server._clusters[idx_2]

        #Select 2 clients from cluster1 and 1 client from cluster2
        client_1a_idx, client_1b_idx = random.sample(cluster_1,2)
        client_2_idx = random.choice(cluster_2)

        client_1a = self.clients[client_1a_idx]
        client_1b = self.clients[client_1b_idx]
        client_2 = self.clients[client_2_idx]

        layer_names_1 = ['layer3','layer4']
        layer_names_2 = ['layer3','layer4']


        #Compute intra-cluster CKA
        model1 =  client_1a._model # Or any neural network of your choice
        model2 = client_1b._model

        intra_cka = CKA(model1, model2,
                model1_name="Model_1a",   # good idea to provide names to avoid confusion
                model2_name="Model_1b",
                # model1_layers=layer_names_1, # List of layers to extract features from
                # model2_layers=layer_names_2, # extracts all layer features by default   
                device=device)

        intra_cka.compare(dataloader) # secondary dataloader is optional
        intra_results = intra_cka.export()  # returns a dict that contains model names, layer names

        #Compute inter-cluster CKA
        model1 =  client_1a._model # Or any neural network of your choice
        model2 = client_2._model

        inter_cka = CKA(model1, model2,
                model1_name="Model_1a",   # good idea to provide names to avoid confusion
                model2_name="Model_2",
                # model1_layers=layer_names_1, # List of layers to extract features from
                # model2_layers=layer_names_2, # extracts all layer features by default    
                device=device)

        inter_cka.compare(dataloader) # secondary dataloader is optional
        inter_results = inter_cka.export()  # returns a dict that contains model names, layer names

        return intra_results, inter_results



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
                
                client.set_model_params(globa_model)
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
