import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
# import wandb


import numpy as np
from tqdm import tqdm
import os
from simple_parsing import Serializable
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd

from learning.dataset.fetching_value_dataset import FetchingQValueDataset, FetchingVValueDataset
from learning.dataset.common import FetchingDatasetConfig
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.value import ValueNet, HistoryPlaceValueonly
from learning.utils import save_checkpoint, AverageMeter, ProgressMeter, get_n_params
from learning.loss import ELBOLoss


@dataclass
class Setting(Serializable):

    # Dataset
    dataname: str = "May6th_3obj_depth8_3000"
    sim_or_exec: str = "sim_dataset"
    file_path_train_entire_annotation: str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/entire.csv"
    data_path_train_dataset_json     : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
    data_path_train_dataset_npz      : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
    file_path_eval_entire_annotation : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/entire.csv"
    data_path_eval_dataset_json      : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_eval_dataset_npz       : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"


    # This parameter governs the model size
    dim_model_hidden = 256
    seq_length = 8
    # Transformer params
    train_q_value    = False
    no_rollout_input = False
    dim_gpt_hidden   = dim_model_hidden
    dim_condition    = dim_model_hidden
    gpt_config: GPT2FetchingConditioner.Config \
        = GPT2FetchingConditioner.Config(
            # Data type
            image_res           = 64,
            dim_obs_rgbd_ch     = 4,    # RGBD
            dim_obs_rgbd_encode = dim_model_hidden,
            dim_obs_grasp       = 1,
            dim_action_input    = 8,
            dim_goal_input      = 5,
            # Architecture
            dim_hidden          = dim_gpt_hidden,
            num_heads           = 2,
            dim_ffn             = dim_model_hidden,
            num_gpt_layers      = 2,
            dropout_rate        = 0.1,
            # Positional encoding
            max_len             = 100,
            seq_len             = seq_length,
            # Output
            dim_condition       = dim_condition)
    # Value head params
    value_config: ValueNet.Config\
        = ValueNet.Config(
            dim_condition = dim_model_hidden)

    # Dataset params
    dataset_config: FetchingDatasetConfig \
        = FetchingDatasetConfig(
            seq_length    = seq_length,
            image_res     = 64)
    
    # Training
    device       : str   = "cuda:1"
    num_workers  : int   = 8
    max_steps    : int   = 200000
    batch_size   : int   = 1024
    learning_rate: float = 0.0001
    milestones   : Tuple[int] = (4000,)

    # Logging 
    exp_dir       : str = '/home/sanghyeon/workspace/POMDP/Simulation/pybullet_env/learning/exp'
    model_name    : str = f'5.10_value_mse_{dataname}_q={train_q_value}_{sim_or_exec}_dim{dim_model_hidden}_batch{batch_size}_lr{learning_rate}_norollout={no_rollout_input}'
    train_log_freq: int = 10
    eval_log_freq : int = 10 
    eval_save_freq: int = 100    # per training step



def main(config: Setting):

    # wandb.init(
    #     project = "history_mse",
    #     config={
    #         "learning_rate": config.learning_rate,
    #         "batch_size": config.batch_size,
    #         "policy": False,
    #         "value": True, 
    #         "type": "MSE"},
    #     sync_tensorboard=True )


    # Train/eval split
    train_entire_annots      = pd.read_csv(config.file_path_train_entire_annotation)["filename"].tolist()
    train_entire_num_subdata = pd.read_csv(config.file_path_train_entire_annotation)["num_subdata"].tolist()
    eval_entire_annots       = pd.read_csv(config.file_path_eval_entire_annotation)["filename"].tolist()
    eval_entire_num_subdata  = pd.read_csv(config.file_path_eval_entire_annotation)["num_subdata"].tolist()
    
    # Dataset
    train_dataset_kwargs = {
        "config"          : config.dataset_config,
        "data_path_json"  : config.data_path_train_dataset_json,
        "data_path_npz"   : config.data_path_train_dataset_npz,
        "annotations"     : train_entire_annots,
        "num_subdata"     : train_entire_num_subdata, 
        "no_rollout_input": config.no_rollout_input}
    eval_dataset_kwargs = {
        "config"          : config.dataset_config,
        "data_path_json"  : config.data_path_eval_dataset_json,
        "data_path_npz"   : config.data_path_eval_dataset_npz,
        "annotations"     : eval_entire_annots,
        "num_subdata"     : eval_entire_num_subdata,
        "no_rollout_input": config.no_rollout_input }
    if config.train_q_value:
        print("Training Q value...")
        train_dataset = FetchingQValueDataset(**train_dataset_kwargs)
        eval_dataset = FetchingQValueDataset(**eval_dataset_kwargs)
    else:
        print("Training V value...")
        train_dataset = FetchingVValueDataset(**train_dataset_kwargs)
        eval_dataset = FetchingVValueDataset(**eval_dataset_kwargs)

    # Sampler
    train_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = train_dataset.weights, 
            num_samples = len(train_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)
    eval_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = eval_dataset.weights, 
            num_samples = len(eval_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)

    # Dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler = train_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler = eval_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)

    # Model
    model = HistoryPlaceValueonly( 
        fetching_gpt_config = config.gpt_config,
        value_config        = config.value_config ).to(config.device)
    optimizer = Optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = Optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000], gamma=0.1)
    print(f"params: {get_n_params(model)}")

    # Metric
    mse_loss_fn = nn.MSELoss()

    # Logger
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)
    logger = SummaryWriter(model_dir)
    # Save configuration
    config.save(model_dir + "/config.yaml")


    # Train/eval routine
    train_wrapper = TrainWrapper(
        config    = config,
        loader    = train_loader,
        model     = model,
        loss_fn   = mse_loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger)
    eval_wrapper = EvalWrapper(
        config    = config,
        loader    = eval_loader,
        model     = model,
        loss_fn   = mse_loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger,
        model_dir = model_dir)
    

    # Steps loop (No epoch here.)
    print("Training started. Checkout tensorboard.")
    for steps in range(0, config.max_steps):        
        # Training
        train_wrapper.train_step_fn(steps)
        # Evaluating
        if steps % config.eval_log_freq == 0:
            eval_wrapper.eval_step_fn(steps)



class TrainWrapper:

    def __init__(self, config: Setting,
                       loader: DataLoader,
                       model: nn.Module,
                       loss_fn: nn.Module,
                       optimizer: Optim.Optimizer,
                       scheduler: Optim.lr_scheduler._LRScheduler,
                       logger: SummaryWriter):
        """1 step train function

        Args:
            config (Setting): Config file
            loader (DataLoader): Dataloader with random sampler
            model (nn.Module): Model
            loss_fn (nn.Module): Loss function
            optimizer (Optim.Optimizer): Optimizer
            scheduler (Optim.lr_scheduler._LRScheduler): Scheduler
            logger (SummaryWriter): Logger
        """
        self.config    = config
        self.loader    = loader
        self.iterator  = iter(loader)
        self.model     = model
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger    = logger
        self.__reset_meters()


    def __reset_meters(self):
        """Reset average meters"""
        self.loss_value_meter   = AverageMeter('Value total', ':.4e')


    def __get_batch_from_dataloader(self) -> Tuple[torch.Tensor]:
        """Only applicable when used with weighted random batch sampler"""
        try:
            data = next(self.iterator)
        except StopIteration:
            print("Restarting data loader iterator")
            self.iterator = iter(self.loader)
            data = next(self.iterator)

        return data


    def train_step_fn(self, steps: int):
        """1 step train function

        Args:
            steps (int): Current step
        """  
        self.model.train()
        
        # Draw a batch
        data = self.__get_batch_from_dataloader()

        # To device...
        init_obs_rgbd, init_obs_grasp, goal, \
            seq_action, seq_obs_rgbd, seq_obs_grasp, \
            action_mask, obs_mask, \
            input_trajectory_length, full_trajectory_length, \
            future_discounted_reward_label, success_or_fail = ([d.to(self.config.device) for d in data])

        # Forward
        pred_value = self.model(
            init_obs_rgbd, init_obs_grasp, goal,
            seq_action, seq_obs_rgbd, seq_obs_grasp,
            action_mask, obs_mask)

        # Loss calculation
        mse_loss: torch.Tensor = self.loss_fn(pred_value, future_discounted_reward_label)

        # Backprop + Optimize ...
        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Logging 1 (per steps)
        self.logger.add_scalar('Loss(MSE)/train_steps', mse_loss.item(), steps)
        # Logging 2 (console)
        self.loss_value_meter.update(mse_loss.item())
        if steps % self.config.train_log_freq == 0:
            print(f"[Train {steps}] {self.loss_value_meter}")
            self.__reset_meters()



class EvalWrapper:

    def __init__(self, config: Setting,
                       loader: DataLoader,
                       model: nn.Module,
                       loss_fn: nn.Module,
                       optimizer: Optim,
                       scheduler: Optim.lr_scheduler._LRScheduler,
                       logger: SummaryWriter,
                       model_dir: str):
        """1 epoch eval function

        Args:
            config (Setting): Config file
            epoch (int): Current epoch
            loader (DataLoader): Data loader
            model (nn.Module): Model
            loss_fn (nn.Module): Loss fn
            optimizer (Optim.Optimizer): Optimizer
            scheduler (Optim.lr_scheduler._LRScheduler): Scheduler
            logger (SummaryWriter): Logger
            model_dir (str): Checkpoint save dir
        """
        self.config     = config
        self.loader     = loader
        self.iterator   = iter(loader)
        self.model      = model
        self.loss_fn    = loss_fn
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.logger     = logger
        self.model_dir  = model_dir
        self.best_error = 10000.
        self.__reset_meters()

    
    def __reset_meters(self):
        """Reset average meters"""
        self.loss_value_meter   = AverageMeter('Value total', ':.4e')


    def __get_batch_from_dataloader(self) -> Tuple[torch.Tensor]:
        """Only applicable when used with weighted random batch sampler"""
        try:
            data = next(self.iterator)
        except StopIteration:
            print("Restarting data loader iterator")
            self.iterator = iter(self.loader)
            data = next(self.iterator)

        return data


    def eval_step_fn(self, steps: int):
        """1 step eval function

        Args:
            steps (int): Current step
        """
        self.model.eval()
        with torch.no_grad():
            if str(self.config.device) == 'cuda':
                torch.cuda.empty_cache()
        
            # Draw a batch
            data = self.__get_batch_from_dataloader()

            # To device...
            init_obs_rgbd, init_obs_grasp, goal, \
                seq_action, seq_obs_rgbd, seq_obs_grasp, \
                action_mask, obs_mask, \
                input_trajectory_length, full_trajectory_length, \
                future_discounted_reward_label, success_or_fail = ([d.to(self.config.device) for d in data])

            # Forward
            pred_value = self.model(
                init_obs_rgbd, init_obs_grasp, goal,
                seq_action, seq_obs_rgbd, seq_obs_grasp,
                action_mask, obs_mask)
            
            # Loss calculation
            mse_loss: torch.Tensor = self.loss_fn(pred_value, future_discounted_reward_label)

        # Logging 1 (tensorboard)
        self.logger.add_scalar('Loss(MSE)/eval', mse_loss.item(), steps)
        # Logging 2 (console)
        self.loss_value_meter.update(mse_loss.item())
        print(f"[Eval  {steps}] {self.loss_value_meter}")
        self.__reset_meters()

        # Save the last model
        if steps % self.config.eval_save_freq == 0:
            save_checkpoint("Saving the last model!",
                            os.path.join(self.model_dir, f"checkpoint{steps}.pth"),
                            -1, mse_loss.item(), 
                            self.model, self.optimizer, self.scheduler)

        # Save the best model
        if mse_loss.item() < self.best_error:
            self.best_error = mse_loss.item()
            save_checkpoint("Saving the best model!",
                            os.path.join(self.model_dir, "best.pth"),
                            -1, self.best_error, 
                            self.model, self.optimizer, self.scheduler)



if __name__=="__main__":
    main(Setting())