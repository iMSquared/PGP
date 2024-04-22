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

from learning.dataset.fetching_value_dataset import FetchingVValueDataset, FetchingSuccessFailDataset
from learning.dataset.common import FetchingDatasetConfig
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.value import ValueNet, HistoryPlaceValueonly
from learning.utils import save_checkpoint, AverageMeter, ProgressMeter, get_n_params
from learning.loss import ELBOLoss


@dataclass
class Setting(Serializable):

    # Dataset
    dataname: str = "May2nd_cuboid12_large"
    sim_or_exec: str = "sim_dataset"
    file_path_train_success_annotation: str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/success.csv"
    file_path_train_fail_annotation        : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/fail.csv"
    data_path_train_dataset_json      : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
    data_path_train_dataset_npz       : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
    file_path_eval_success_annotation : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/eval/success.csv"
    file_path_eval_fail_annotation         : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/eval/fail.csv"
    data_path_eval_dataset_json       : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_eval_dataset_npz        : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"


    # This parameter governs the model size
    dim_model_hidden = 256
    seq_length = 6
    # Transformer params
    dim_gpt_hidden = dim_model_hidden
    dim_condition  = dim_model_hidden
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
    device       : str        = "cuda:1"
    num_workers  : int        = 4
    max_steps    : int        = 200000
    batch_size   : int        = 1024
    learning_rate: float      = 0.0001
    milestones   : Tuple[int] = (4000,)

    # Logging 
    exp_dir         : str = '/home/jiyong/workspace/POMDP/Simulation/pybullet_env/learning/exp'
    # model_name      : str = 'test'
    model_name      : str = f'5.9_value_success_fail_{dataname}_q=False_{sim_or_exec}_dim{dim_model_hidden}_batch{batch_size}_lr{learning_rate}'
    train_log_freq  : int = 10
    eval_num_samples: int = 5000    # Minimum number of samples to use for evaluation metric.
    eval_log_freq   : int = 50
    eval_save_freq  : int = eval_log_freq     # per training step


def main(config: Setting):

    # wandb.init(
    #     project = "history_preference",
    #     config={
    #         "learning_rate": config.learning_rate,
    #         "batch_size": config.batch_size,
    #         "policy": False,
    #         "value": True, 
    #         "type": "Preference"},
    #     sync_tensorboard=True )


    # Train/eval split
    train_success_annots      = pd.read_csv(config.file_path_train_success_annotation)["filename"].tolist()
    train_success_num_subdata = pd.read_csv(config.file_path_train_success_annotation)["num_subdata"].tolist()
    train_fail_annots       = pd.read_csv(config.file_path_train_fail_annotation)["filename"].tolist()
    train_fail_num_subdata  = pd.read_csv(config.file_path_train_fail_annotation)["num_subdata"].tolist()
    eval_success_annots       = pd.read_csv(config.file_path_eval_success_annotation)["filename"].tolist()
    eval_success_num_subdata  = pd.read_csv(config.file_path_eval_success_annotation)["num_subdata"].tolist()
    eval_fail_annots        = pd.read_csv(config.file_path_eval_fail_annotation)["filename"].tolist()
    eval_fail_num_subdata   = pd.read_csv(config.file_path_eval_fail_annotation)["num_subdata"].tolist()
    
    # Dataset
    success_train_dataset_kwargs = {
        "config"        : config.dataset_config,
        "data_path_json": config.data_path_train_dataset_json,
        "data_path_npz" : config.data_path_train_dataset_npz,
        "annotations"   : train_success_annots,
        "num_subdata"   : train_success_num_subdata }
    success_eval_dataset_kwargs = {
        "config"        : config.dataset_config,
        "data_path_json": config.data_path_eval_dataset_json,
        "data_path_npz" : config.data_path_eval_dataset_npz,
        "annotations"   : eval_success_annots,
        "num_subdata"   : eval_success_num_subdata }
    fail_train_dataset_kwargs = {
        "config"        : config.dataset_config,
        "data_path_json": config.data_path_train_dataset_json,
        "data_path_npz" : config.data_path_train_dataset_npz,
        "annotations"   : train_fail_annots,
        "num_subdata"   : train_fail_num_subdata }
    fail_eval_dataset_kwargs = {
        "config"        : config.dataset_config,
        "data_path_json": config.data_path_eval_dataset_json,
        "data_path_npz" : config.data_path_eval_dataset_npz,
        "annotations"   : eval_fail_annots,
        "num_subdata"   : eval_fail_num_subdata }
    
    success_train_dataset = FetchingVValueDataset(**success_train_dataset_kwargs)
    success_eval_dataset  = FetchingVValueDataset(**success_eval_dataset_kwargs)
    fail_train_dataset  = FetchingVValueDataset(**fail_train_dataset_kwargs)
    fail_eval_dataset   = FetchingVValueDataset(**fail_eval_dataset_kwargs)
        
    success_fail_train_dataset = FetchingSuccessFailDataset(
        success_dataset    = success_train_dataset,
        comparison_dataset = fail_train_dataset)
    success_fail_eval_dataset = FetchingSuccessFailDataset(
        success_dataset    = success_eval_dataset,
        comparison_dataset = fail_eval_dataset)

    # Sampler
    train_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = success_fail_train_dataset.weights, 
            num_samples = len(success_fail_train_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)
    eval_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = success_fail_eval_dataset.weights, 
            num_samples = len(success_fail_eval_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)

    # Dataloader
    train_loader = DataLoader(
        success_fail_train_dataset,
        batch_sampler = train_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)
    eval_loader = DataLoader(
        success_fail_eval_dataset,
        batch_sampler = eval_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)


    # Model
    model = HistoryPlaceValueonly( 
        fetching_gpt_config = config.gpt_config,
        value_config        = config.value_config ).to(config.device)
    optimizer = Optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = Optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    print(f"params: {get_n_params(model)}")

    # Metric
    bce_loss_fn = nn.BCELoss()

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
        loss_fn   = bce_loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger)
    eval_wrapper = EvalWrapper(
        config    = config,
        loader    = eval_loader,
        model     = model,
        loss_fn   = bce_loss_fn,
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
        self.loss_bce_meter = AverageMeter('Value total', ':.4e')
        self.accuracy_meter = AverageMeter('Accuracy', ':.4e')


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
        (success_init_observation_rgbd,
            success_init_observation_grasp,
            success_goal,
            success_seq_action,
            success_seq_observation_rgbd,
            success_seq_observation_grasp,
            success_action_mask,
            success_observation_mask,
            success_input_trajectory_length,
            success_full_trajectory_length,
            success_future_discounted_reward_label,
            success_success_or_fail,
            comparison_init_observation_rgbd,
            comparison_init_observation_grasp,
            comparison_goal,
            comparison_seq_action,
            comparison_seq_observation_rgbd,
            comparison_seq_observation_grasp,
            comparison_action_mask,
            comparison_observation_mask,
            comparison_input_trajectory_length,
            comparison_full_trajectory_length,
            comparison_future_discounted_reward_label,
            comparison_success_or_fail,
            preference, 
            is_equal) = ([d.to(self.config.device) for d in data])


        # Forward shape=(B)
        pred_success_preference = self.model(
            success_init_observation_rgbd, success_init_observation_grasp, success_goal, 
            success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp,
            success_action_mask, success_observation_mask)
        pred_comparison_preference = self.model(
            comparison_init_observation_rgbd, comparison_init_observation_grasp, comparison_goal, 
            comparison_seq_action, comparison_seq_observation_rgbd, comparison_seq_observation_grasp,
            comparison_action_mask, comparison_observation_mask)
        pred_concat = torch.cat((pred_success_preference, pred_comparison_preference), dim=-1)
        pred_concat = torch.softmax(pred_concat, dim=-1)
                
        # Eval 1. Loss calculation
        loss_total: torch.Tensor = self.loss_fn(pred_concat, preference)
        # Eval 2. Accuracy
        is_equal = is_equal.squeeze(1)
        pred_argmax  = torch.argmax(pred_concat, dim=1)
        label_argmax = torch.argmax(preference, dim=1)
        num_samples  = torch.sum(~is_equal)
        num_correct  = torch.sum(torch.eq(pred_argmax[~is_equal], label_argmax[~is_equal]))
        batch_accuracy = float(num_correct)/float(num_samples)

        # Backprop + Optimize ...
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Log loss
        self.loss_bce_meter.update(loss_total.item())
        self.accuracy_meter.update(batch_accuracy)
        # Logging 1 (per steps)
        self.logger.add_scalar('Loss(Preference)/train_steps', loss_total.item(), steps)
        self.logger.add_scalar('Accuracy(Preference)/train_steps', batch_accuracy, steps)
        # Logging 2 (console)
        if steps % self.config.train_log_freq == 0:
            print(f"[Train {steps}] {self.loss_bce_meter}, {self.accuracy_meter}")
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
        self.config        = config
        self.loader        = loader
        self.iterator      = iter(loader)
        self.model         = model
        self.loss_fn       = loss_fn
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.logger        = logger
        self.model_dir     = model_dir
        self.best_accuracy = 0.
        self.__reset_meters()

    
    def __reset_meters(self):
        """Reset average meters"""
        self.loss_bce_meter = AverageMeter('Value total', ':.4e')
        self.accuracy_meter = AverageMeter('Accuracy', ':.4e')
        

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
            # Repeat evaluation until get enough samples
            num_samples_total = 0
            while num_samples_total < self.config.eval_num_samples:
                # Draw a batch
                data = self.__get_batch_from_dataloader()

                # To device...
                (success_init_observation_rgbd,
                    success_init_observation_grasp,
                    success_goal,
                    success_seq_action,
                    success_seq_observation_rgbd,
                    success_seq_observation_grasp,
                    success_action_mask,
                    success_observation_mask,
                    success_input_trajectory_length,
                    success_full_trajectory_length,
                    success_future_discounted_reward_label,
                    success_success_or_fail,
                    comparison_init_observation_rgbd,
                    comparison_init_observation_grasp,
                    comparison_goal,
                    comparison_seq_action,
                    comparison_seq_observation_rgbd,
                    comparison_seq_observation_grasp,
                    comparison_action_mask,
                    comparison_observation_mask,
                    comparison_input_trajectory_length,
                    comparison_full_trajectory_length,
                    comparison_future_discounted_reward_label,
                    comparison_success_or_fail,
                    preference, 
                    is_equal) = ([d.to(self.config.device) for d in data])


                # Forward shape=(B)
                pred_success_preference = self.model(
                    success_init_observation_rgbd, success_init_observation_grasp, success_goal, 
                    success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp,
                    success_action_mask, success_observation_mask)
                pred_comparison_preference = self.model(
                    comparison_init_observation_rgbd, comparison_init_observation_grasp, comparison_goal, 
                    comparison_seq_action, comparison_seq_observation_rgbd, comparison_seq_observation_grasp,
                    comparison_action_mask, comparison_observation_mask)
                pred_concat = torch.cat((pred_success_preference, pred_comparison_preference), dim=-1)
                pred_concat = torch.softmax(pred_concat, dim=-1)

                # Eval 1. Loss
                loss_total: torch.Tensor = self.loss_fn(pred_concat, preference)
                # Eval 2. Accuracy
                is_equal = is_equal.squeeze(1)
                pred_argmax  = torch.argmax(pred_concat, dim=1)
                label_argmax = torch.argmax(preference, dim=1)
                num_samples  = torch.sum(~is_equal)
                num_correct  = torch.sum(torch.eq(pred_argmax[~is_equal], label_argmax[~is_equal]))
                batch_accuracy = float(num_correct)/float(num_samples)

                # Log loss
                self.loss_bce_meter.update(loss_total.item())
                self.accuracy_meter.update(batch_accuracy, n=num_samples)

                # Repeat evaluation...
                num_samples_total += num_samples


        # Logging 1 (tensorboard)
        self.logger.add_scalar('Loss(Preference)/eval', self.loss_bce_meter.avg, steps)
        self.logger.add_scalar('Accuracy(Preference)/eval', self.accuracy_meter.avg, steps)
        # Logging 2 (console)
        print(f"[Eval  {steps}] {self.loss_bce_meter}, {self.accuracy_meter}")
        last_avg_accuracy = self.accuracy_meter.avg
        self.__reset_meters()


        # Save the last model
        if steps % self.config.eval_save_freq == 0:
            save_checkpoint("Saving the last model!",
                            os.path.join(self.model_dir, f"checkpoint{steps}.pth"),
                            -1, loss_total.item(), 
                            self.model, self.optimizer, self.scheduler)

        # Save the best model
        if last_avg_accuracy > self.best_accuracy:
            self.best_accuracy = last_avg_accuracy
            save_checkpoint("Saving the best model!",
                            os.path.join(self.model_dir, "best.pth"),
                            -1, self.best_accuracy, 
                            self.model, self.optimizer, self.scheduler)




if __name__=="__main__":
    main(Setting())