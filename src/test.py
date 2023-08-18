from torch_cka import CKA
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

transform = ToTensor()
cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform)


model1 = models.resnet18(pretrained=True)  # Or any neural network of your choice
model2 = models.resnet18(pretrained=False)

layer_names_resnet1 = ['layer3','layer4']
layer_names_resnet2 = ['layer3','layer4']

dataloader = DataLoader(cifar_test, 
                        batch_size=32, # according to your device memory
                        shuffle=False)  # Don't forget to seed your dataloader

cka = CKA(model1, model2,
          model1_name="ResNet18",   # good idea to provide names to avoid confusion
          model2_name="ResNet18_2",   
        #   model1_layers=layer_names_resnet1, # List of layers to extract features from
        #   model2_layers=layer_names_resnet2, # extracts all layer features by default
          device='cuda')

cka.compare(dataloader) # secondary dataloader is optional

results = cka.export()  # returns a dict that contains model names, layer names

plt.imshow(results['CKA'])
plt.xlabel('model_1a')
plt.ylabel('model_1b')
plt.colorbar()
plt.savefig('test-cka.png')
plt.close()