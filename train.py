import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from data import create_dataset, create_loader
from torch.utils.data import DataLoader
from configs import get_crop_ratio
from net.M2Restore import M2Restore
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = M2Restore(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (inputs, targets, task_idx) = batch
        restored, loss_b = self.net(inputs)

        loss = self.loss_fn(restored, targets) + 0.01*loss_b
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=180)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    seed=42
    # random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="M2Restore-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)

    crop_ratio = get_crop_ratio(dataset_name=opt.dataset)

    trainset = create_dataset(augment_enable=False,
                              task_name=opt.task,
                              ds_name=opt.dataset,
                              split='train',
                              random_crop_ratio=crop_ratio)

    trainloader = create_loader(dataset=trainset,
                                ds_name=opt.dataset,
                                task_name=opt.task,
                                batch_size=opt.batch_size,
                                global_seed=opt.global_seed,
                                shuffle=True,
                                num_workers=opt.num_workers)
    
    model = IRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,
                          accelerator="gpu",
                          devices=opt.num_gpus,
                          logger=logger,
                          callbacks=[checkpoint_callback],
                          accumulate_grad_batches = 4,
                          )
    trainer.fit(model=model,
                train_dataloaders=trainloader,
                ckpt_path = '/home/ubuntu/M2/last.ckpt'
    )

if __name__ == '__main__':
    main()