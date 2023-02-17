import torch
import torch.nn as nn
# TODO Max, Avg, Majority vote(hard max [1,0,0,0])
import torch.nn.functional as F

def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)

class AvgEnsemble(nn.Module):
    def __init__(self, net_list):
        super(AvgEnsemble, self).__init__()
        self.estimators = nn.ModuleList(net_list)

    def forward(self, x):
        outputs = [
            F.softmax(estimator(x), dim=1) for estimator in self.estimators
        ]
        proba = average(outputs)

        return proba





class AvgEnsemble1(nn.Module):
    def __init__(self, nets: list):
        super(AvgEnsemble, self).__init__()
        self.nets = nets

    def forward(self, x):
        for i, net in enumerate(self.nets):
            if i == 0:
                res = net(x).unsqueeze(1)
            else:
                res = torch.cat((res, net(x).unsqueeze(1)), 1)
        return torch.mean(res, 1).values


class MaxEnsemble(nn.Module):
    def __init__(self, nets: list):
        super(MaxEnsemble, self).__init__()
        self.nets = nets

    def forward(self, x):

        for i, net in enumerate(self.nets):
            if i == 0:
                res = net(x).unsqueeze(1)
            else:
                res = torch.cat((res, net(x).unsqueeze(1)), 1)
        return torch.max(res, 1).values


class VoteEnsemble(nn.Module):
    def __init__(self, nets: list):
        super(VoteEnsemble, self).__init__()
        self.nets = nets

    def forward(self, x):
        res = []
        for net in self.nets:
            res.append(net(x))
        return nn.Softmax(res)


if __name__ == '__main__':
    from src.RaFL.networks.vgg import vgg11

    net = vgg11()
    net1 = vgg11()
    net2 = vgg11()
    netlist = [net, net2, net1]
    nets = AvgEnsemble(netlist)
    y = nets(torch.randn(128, 3, 32, 32))

    print(y.shape)
