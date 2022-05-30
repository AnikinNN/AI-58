import torch
import torchvision.models as models
import torch.nn as nn

from regressor_on_resnet.mish import Mish


class ResnetRegressor(torch.nn.Module):
    def __init__(self):
        super(ResnetRegressor, self).__init__()
        self.resnet = models.resnet152(pretrained=True, progress=False)
        self.set_train_convolutional_part(False)
        self.resnet.fc = torch.nn.Identity()

        # todo in_features=2049 replace by something like
        # self.resnet.fc.output_shape + 1
        self.fully_connected = torch.nn.Sequential(nn.Linear(in_features=2049, out_features=512),
                                                   Mish(),
                                                   nn.Linear(in_features=512, out_features=128),
                                                   Mish(),
                                                   nn.Linear(in_features=128, out_features=1),
                                                   Mish())

    def forward(self, image_batch, elevation_batch):
        features = self.resnet(image_batch)
        result = self.fully_connected(torch.cat((features, elevation_batch), dim=1))
        return result

    def set_train_convolutional_part(self, value: bool):
        for param in self.resnet.parameters():
            param.requires_grad = value
