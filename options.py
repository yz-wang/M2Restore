import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=4,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')


parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=42, help='number of workers.')

# dataset
parser.add_argument('--dataset', default='allweather', type=str, help='dataset name')
parser.add_argument('--task', default='low_level', type=str, help='task name')
parser.add_argument('--global-seed', default=42, type=int, help='random global seed')

# path
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/type11/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="M2Restore",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="M2Restore",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1, help = "Number of GPUs to use for training")

options = parser.parse_args()

