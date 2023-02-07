import torch

def adjust_net_arc(net, out_features=10):
    layer_name = None
    last_layer = None
    for name, layer in net.named_modules():
        if isinstance(layer, torch.nn.Linear):# the last layer should be a linear layer
            layer_name = name
            last_layer = layer

    predictor = torch.nn.Linear(last_layer.in_features, out_features)
    setattr(net, layer_name.split('.')[0], predictor)

    
    return net