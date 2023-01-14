import torch

from regressor_on_resnet.batch_factory import BatchFactory
from regressor_on_resnet.flux_dataset import FluxDataset
from regressor_on_resnet.nn_logging import Logger
from regressor_on_resnet.metadata_loader import MetadataLoader
from regressor_on_resnet.resnet_regressor import ResnetRegressor
from regressor_on_resnet.sgdr_restarts_warmup import CosineAnnealingWarmupRestarts
from regressor_on_resnet.train_common import train_model

learning_rate = 20e-5
batch_size = 32
cuda_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batch_fields = ['images',
                'masks',
                'fluxes',
                'elevations']
output_size = (512, 512)

logger = Logger()

metadata_loader = MetadataLoader(('./../cloud_applications_v2/expeditions_configs/AI-58-config.json',
                                  './../cloud_applications_v2/expeditions_configs/AMK-79-config.json',
                                  './../cloud_applications_v2/expeditions_configs/ABP-42-config.json',
                                  './../cloud_applications_v2/expeditions_configs/AI-52-config.json',
                                  ),
                                 radiation_threshold=10,
                                 split=(0.6, 0.2, 0.2),
                                 store_path=logger.misc_dir)

train_dataset, val_dataset = [
    FluxDataset(flux_frame=i,
                batch_size=batch_size,
                do_shuffle=True,
                output_size=output_size,
                batch_fields=batch_fields)
    for i in (metadata_loader.train, metadata_loader.validation)]

train_batch_factory = BatchFactory(dataset=train_dataset,
                                   cuda_device=cuda_device,
                                   do_augment=True,
                                   cpu_queue_length=4,
                                   cuda_queue_length=4,
                                   preprocess_worker_number=15,
                                   cuda_feeder_number=1,
                                   to_variable=True, )

validation_batch_factory = BatchFactory(dataset=val_dataset,
                                        cuda_device=cuda_device,
                                        do_augment=False,
                                        cpu_queue_length=4,
                                        cuda_queue_length=4,
                                        preprocess_worker_number=15,
                                        cuda_feeder_number=1,
                                        to_variable=False, )

model = ResnetRegressor()
model.to(cuda_device)

steps_per_epoch_train = 10  # 1024

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                             first_cycle_steps=32 * steps_per_epoch_train,
                                             cycle_mult=1,
                                             max_lr=learning_rate,
                                             min_lr=5e-7,
                                             warmup_steps=0,
                                             gamma=0.8,
                                             last_epoch=-1)
train_model(
    model=model,
    loss=loss,
    train_batch_factory=train_batch_factory,
    validation_batch_factory=validation_batch_factory,
    logger=logger,
    max_epochs=4,
    steps_per_epoch_train=steps_per_epoch_train,
    steps_per_epoch_valid=steps_per_epoch_train // 2,
    train_convolutional_since_epoch=128,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
)
print()
