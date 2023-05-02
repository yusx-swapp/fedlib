import random


def stratified_cluster_sampler(n_clients, clusters, sample_rate):
    # select clients
    print("clusters to sample from:", clusters)
    assert n_clients == sum([len(cluster) for _,cluster in clusters.items()])
    selected = []
    for _,cluster in clusters.items():
        size = round(sample_rate * (len(cluster)))
        size = max(1,size)
        random.shuffle(cluster)
        #print("cluster:",cluster,"\tsize:",size)
        selected += cluster[:size]
    
    print("Actual sample size:", len(selected),"\tExpected sample size:", int(sample_rate * n_clients))
    #assert abs(len(selected) - int(sample_rate * n_clients)) in [0,1]

    return selected
