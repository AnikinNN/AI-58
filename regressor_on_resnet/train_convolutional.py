import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))

import torch

from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.train_common import train_model
from regressor_on_resnet.resnet_regressor import ResnetRegressor
from regressor_on_resnet.pretrained_loader import PretrainedLoader


logger = Logger()

base_run_number = 115

cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

batch_size = 64

pretrained_loader = PretrainedLoader()
pretrained_loader.init_using_logger(logger, base_run_number)

train_set = FluxDataset(flux_frame=pretrained_loader.train,
                        batch_size=batch_size,
                        do_shuffle=True)

val_set = FluxDataset(flux_frame=pretrained_loader.validation,
                      batch_size=batch_size,
                      do_shuffle=True)

hard_mining_train_set = FluxDataset(flux_frame=pretrained_loader.train,
                                    batch_size=batch_size,
                                    do_shuffle=False)

modified_resnet = pretrained_loader.model
modified_resnet.set_train_convolutional_part(True)
modified_resnet.to(cuda_device)

train_model(modified_resnet,
            train_dataset=train_set,
            val_dataset=val_set,
            hard_mining_dataset=hard_mining_train_set,
            logger=logger,
            cuda_device=cuda_device,
            max_epochs=256,
            use_warmup=True)

print()
