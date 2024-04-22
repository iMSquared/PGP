import os
import shutil
import glob
import time
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import List
import pickle
import torch as th
import numpy as np
from tensorboardX import SummaryWriter
# import wandb

from load import get_loader
from model import GPT2, RNN, LSTM, CVAE, ValueNet, ValueNetDiscreteRepresentation
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss, CQLLoss, RegressionLossValueWithNegativeData
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients
from run import Settings


src_dir = "/home/jiyong/workspace/POMDP/toy_domain/Learning/dataset/sim_3.25/fail_mini"
dst_dir = "/home/jiyong/workspace/POMDP/toy_domain/Learning/dataset/sim_3.25/out"
filt = 50

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
    
config = Settings()
dataset = glob.glob(f'{src_dir}/*.pickle')

dataset = os.listdir(src_dir)
num_data_move = int(len(dataset) * ratio)
print(num_data_move)

for d in dataset[-num_data_move:]:
    data_path = os.path.join(src_dir, d)
    shutil.move(data_path, dst_dir)