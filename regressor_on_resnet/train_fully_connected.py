import os
import sys

sys.path.append(os.path.join(sys.path[0], '..'))

import torch
from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.metadata_loader import MetadataLoader
from regressor_on_resnet.resnet_regressor import ResnetRegressor
from regressor_on_resnet.train_common import train_model

logger = Logger()

cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

metadata_loader = MetadataLoader(('./../cloud_applications_v2/expeditions_configs/AI-58-config.json',
                                  './../cloud_applications_v2/expeditions_configs/AMK-79-config.json',
                                  './../cloud_applications_v2/expeditions_configs/ABP-42-config.json',
                                  './../cloud_applications_v2/expeditions_configs/AI-52-config.json',
                                  ),
                                 radiation_threshold=10,
                                 split=(0.6, 0.2, 0.2),
                                 store_path=logger.misc_dir)
batch_size = 64
train_set = FluxDataset(flux_frame=metadata_loader.train,
                        batch_size=batch_size,
                        do_shuffle=True,
                        do_augment=True)

val_set = FluxDataset(flux_frame=metadata_loader.validation,
                      batch_size=batch_size,
                      do_shuffle=True,
                      do_augment=False)

modified_resnet = ResnetRegressor()
modified_resnet.to(cuda_device)

train_model(modified_resnet,
            train_dataset=train_set,
            val_dataset=val_set,
            logger=logger,
            cuda_device=cuda_device,
            max_epochs=250)

print()
