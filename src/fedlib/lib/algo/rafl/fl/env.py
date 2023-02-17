from cmath import inf
from copy import deepcopy
from typing import List
from ..nas import adjust_net_arc, ArcSampler

def init_fl(resource_constraints:List[float], supernet_name:str="ofa_supernet_resnet50", tolerance=1000, max_try=10000, image_size:int=32, num_classes=10):
    """
    """
    n_clients = len(resource_constraints)
    nets = {net_i: None for net_i in range(n_clients)}
    nets_MACs = []

    knwl_net = None
    knwl_net_MACs = inf

    arc_sampler = ArcSampler(supernet_name=supernet_name, image_size=image_size)
    for net_i, MACs in enumerate(resource_constraints):
        
        net, net_MACs = arc_sampler.sampling_arc(MACs=MACs,tolerance=tolerance,max_try=max_try)

        #@Phuong: change the network architecture: remember to add parameter num_classes also in init_fl!!

        net = adjust_net_arc(net, num_classes)
        
        nets[net_i] = net
        nets_MACs.append(net_MACs)

        if net_MACs < knwl_net_MACs: # smallest model to be the knowledge network
            knwl_net = deepcopy(net)

    return nets, knwl_net, nets_MACs

def init_client_env():
    
    #TODO: return optimizer lr_scheduler ...
    pass

def init_cloud_env():

    pass
