import torch
import torchvision.models as models
import torch.nn as nn

from regressor_on_resnet.mish import Mish


class ResnetBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True, progress=False)
        self.set_train_convolutional_part(False)
        self.resnet.fc = torch.nn.Identity()
        self.widths: list = None
        self.tail: torch.nn.Sequential = None

    def init_tail(self, output_num, widths, last_activation: bool):
        # self.resnet.output_shape + 1
        # rely on resnet50 inner structure
        self.widths = [self.resnet.layer4[2].conv3.out_channels + 1]
        self.widths.extend(widths)
        self.widths.append(output_num)

        layers = []

        for i in range(len(self.widths) - 1):
            layers.append(nn.Linear(in_features=self.widths[i], out_features=self.widths[i + 1]))
            layers.append(Mish())

        if not last_activation:
            layers = layers[:-1]

        self.tail = torch.nn.Sequential(*layers)

    def forward(self, image_batch: torch.Tensor, elevation_batch: torch.Tensor):
        features = self.resnet(image_batch)
        result = self.tail(torch.cat((features, elevation_batch), dim=1))
        return result

    def set_train_convolutional_part(self, value: bool):
        for param in self.resnet.parameters():
            param.requires_grad = value


class ResnetRegressor(ResnetBase):
    def __init__(self, widths: tuple = (512, 128)):
        super().__init__()
        self.init_tail(1, widths, True)


class ResnetClassifier(ResnetBase):
    def __init__(self, widths: tuple, class_number: int):
        super().__init__()
        self.init_tail(class_number, widths, False)
