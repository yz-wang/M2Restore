import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data import create_dataset
from utils.val_utils import validation
from net.M2Restore import M2Restore
import torch.optim as optim
from utils.schedulers import LinearWarmupCosineAnnealingLR

class IRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = M2Restore(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (inputs, targets, task_idx) = batch
        restored = self.net(inputs)

        loss = self.loss_fn(restored, targets)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)

        return [optimizer], [scheduler]

def test_restore(net, dataset, task="derain"):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    net.eval()
    print('--- Testing starts! ---')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = task
    val_psnr, val_ssim = validation(net, testloader, device, exp_name, save_tag=False)
    print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))

def test_CDD11(net, dataset, task="CDD11"):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    net.eval()
    print('--- Testing starts! ---')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = task
    val_psnr, val_ssim = validation(net, testloader, device, exp_name)
    print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--task', default='deraindrop', type=str, help='task name')
    parser.add_argument('--dataset', default='allweather', type=str, help='dataset name')
    parser.add_argument('--split', default='test', type=str, help='dataset split')
    parser.add_argument('--output_path', type=str, default="M2Restore_results/", help='output save path')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default="/home/ubuntu/M2Restore/ckpt/M2Restore.ckpt",
                        help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = testopt.ckpt_path

    testset = create_dataset(task_name=testopt.task,
                             ds_name=testopt.dataset,
                             split=testopt.split,
                             random_crop_ratio=(1.0, 1.0))

    print("CKPT name : {}".format(ckpt_path))

    net = IRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.dataset == 'cdd11':
        print('Start testing CDD11 streak removal...')
        test_CDD11(net, testset, task="CDD11")
    else:
        if testopt.task == 'derain':
            print('Start testing rain streak removal...')
            test_restore(net, testset, task="derain")
        if testopt.task == 'deraindrop':
            print('Start testing raindrop removal...')
            test_restore(net, testset, task="deraindrop")
        if testopt.task == 'desnow':
            print('Start testing snow streak removal...')
            test_restore(net, testset, task="desnow")