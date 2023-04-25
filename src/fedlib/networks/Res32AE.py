import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models



class ResNet32Autoencoder(nn.Module):
    def __init__(self, embedding_dim=256,n_classes=10):
        super(ResNet32Autoencoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = models.resnet34(pretrained=False) # load ResNet18 pretrained weights
        self.encoder.fc = nn.Identity() # remove the classification layer
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # use adaptive pooling to output a 1x1 feature map
        self.embedding = nn.Linear(512, self.embedding_dim) # embedding layer
        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=n_classes, bias=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=3, padding=1),
            nn.Tanh()
        ) # decoder architecture


    def decoder_forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.view(x.size(0), self.embedding_dim, 1, 1)
        x = self.decoder(x)
        return x
    
    def predictor_forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        pred = self.predictor(x)
        
        return pred

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        pred = self.predictor(x)
        x = x.view(x.size(0), self.embedding_dim, 1, 1)
        x_ = self.decoder(x)

        return pred, x_


class ResNet32Segmentation(nn.Module):
    def __init__(self, embedding_dim=256, n_classes=2):
        super(ResNet32Segmentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder = models.resnet34(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(512, self.embedding_dim)
        self.segmentation = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, n_classes, kernel_size=4, stride=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.view(x.size(0), self.embedding_dim, 1, 1)
        x = self.segmentation(x)
        return x
