import torch
from .efficiency_predictor import *


class ArcSampler:

    def __init__(self, supernet_name: str, image_size: int = 32):
        """
        :param supernet_name:
        :param image_size: image size of input dataset, \
                           used to calculate FLOPs/MACs. \
                           Defualt to 32 (CIFAR-10 image size)
        :return:
        """
        if supernet_name not in ['ofa_supernet_mbv3_w10', 'ofa_supernet_mbv3_w12',
                                 'ofa_supernet_proxyless', 'ofa_supernet_resnet50']:
            raise ValueError("Invalid SuperNet Name. (must in ofa_supernet_mbv3_w10, \
                                ofa_supernet_mbv3_w12, ofa_supernet_proxyless, \
                                ofa_supernet_resnet50)")

        self._supernet_name = supernet_name
        self._supernet = self._initialize_supernet()
        self._image_size = 32

    def _initialize_supernet(self):
        """

        :return:
        """
        super_net = torch.hub.load('mit-han-lab/once-for-all', self._supernet_name, pretrained=True).eval()
        return super_net

    def update_supernet(self, supernet_name):
        """

        :param supernet_name:
        :return:
        """
        if supernet_name not in ['ofa_supernet_mbv3_w10', 'ofa_supernet_mbv3_w12',
                                 'ofa_supernet_proxyless', 'ofa_supernet_resnet50']:
            raise ValueError("Invalid SuperNet Name. (must in ofa_supernet_mbv3_w10, \
                                ofa_supernet_mbv3_w12, ofa_supernet_proxyless, \
                                ofa_supernet_resnet50)")

        self._supernet_name = supernet_name
        self._supernet = self._initialize_supernet()

    def update_imagesize(self, image_size):
        """

        :param image_size:
        :return:
        """
        self._image_size = image_size

    def sampling_arc(self, MACs: float, tolerance: float = 1000, max_try: int = 1000) -> torch.nn.Module:
        """

        :param MACs:
        :param tolerance:
        :param max_try:
        :return:
        """
        efficiency_predictor = self._get_MACs_predictor()
        for i in range(0, max_try):
            net_arc = self._supernet.sample_active_subnet()
            net_arc.update({'image_size': self._image_size})
            net_MACs = efficiency_predictor.get_efficiency(net_arc)
            if MACs - tolerance <= net_MACs <= MACs:
                net = self._supernet.get_active_subnet(preserve_weight=True)
                return net, net_MACs

        raise RuntimeError("Cannot find target architecture within max_try!")

    def _get_MACs_predictor(self):
        """

        :return:
        """
        if self._supernet_name in ['ofa_supernet_mbv3_w10', 'ofa_supernet_mbv3_w12']:
            return Mbv3FLOPsModel(self._supernet)

        elif self._supernet_name in ['ofa_supernet_resnet50']:
            return ResNet50FLOPsModel(self._supernet)

        elif self._supernet_name in ['ofa_supernet_proxyless']:
            return ProxylessNASFLOPsModel(self._supernet)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    arc_sampler = ArcSampler('ofa_supernet_mbv3_w12')
    net, net_MACs = arc_sampler.sampling_arc(100)
    print(net_MACs)
'''
['ofa_net', 'ofa_specialized', 'ofa_specialized_get', 'ofa_supernet_mbv3_w10', 'ofa_supernet_mbv3_w12', 'ofa_supernet_proxyless', 'ofa_supernet_resnet50', 'partial', 'resnet50D_MAC_0_6B', 'resnet50D_MAC_0_9B', 'resnet50D_MAC_1_2B', 'resnet50D_MAC_1_8B', 'resnet50D_MAC_2_4B', 'resnet50D_MAC_3_0B', 'resnet50D_MAC_3_7B', 'resnet50D_MAC_4_1B']
'''


