import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
import wandb


import numpy as np
from tqdm import tqdm
import os
from simple_parsing import Serializable
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd

from learning.dataset.fetching_valueonly_dataset import FetchingValueonlyDataset, FetchingPreferenceValueonlyDataset
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.value import ValueNet, HistoryPlaceValueonly
from learning.utils import save_checkpoint, AverageMeter, ProgressMeter, get_n_params
from learning.loss import ELBOLoss


@dataclass
class Setting(Serializable):

    # Dataset
    dataname: str = "April12th"
    sim_or_exec: str = "sim_dataset"
    file_path_train_success_annotation: str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/success.csv"
    file_path_train_fail_annotation   : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/fail.csv"
    data_path_train_dataset_json      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
    data_path_train_dataset_npz       : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
    file_path_eval_success_annotation : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/success.csv"
    file_path_eval_fail_annotation    : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/fail.csv"
    data_path_eval_dataset_json       : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_eval_dataset_npz        : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"


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
            dim_action_input    = 5,
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
    dataset_config: FetchingValueonlyDataset.Config \
        = FetchingValueonlyDataset.Config(
            train_q_value = True,
            seq_length    = seq_length,
            image_res     = 64)
    

    # Training
    device         : str   = "cuda:1"
    num_workers    : int   = 8
    epochs         : int   = 5000
    batch_size     : int   = 1024
    learning_rate  : float = 0.00025

    # Logging 
    exp_dir        : str   = '/home/sanghyeon/workspace/POMDP/Simulation/pybullet_env/learning/exp'
    model_name     : str   = f'4.13_value_{dataname}_{sim_or_exec}_dim{dim_model_hidden}_ranking_batch{batch_size}_lr{learning_rate}'

    print_freq     : int   = 10     # per training step
    train_eval_freq: int   = 100    # per training step
    eval_freq      : int   = 1      # per epoch



def main(config: Setting):

    # wandb.init(
    #     project = "history_ranking",
    #     config={
    #         "learning_rate": config.learning_rate,
    #         "batch_size": config.batch_size,
    #         "policy": False,
    #         "value": True, 
    #         "type": "ranking"},
    #     sync_tensorboard=True )


    # Train/eval split
    train_success_annots      = pd.read_csv(config.file_path_train_success_annotation)["filename"].tolist()
    train_success_num_subdata = pd.read_csv(config.file_path_train_success_annotation)["num_subdata"].tolist()
    train_fail_annots         = pd.read_csv(config.file_path_train_fail_annotation)["filename"].tolist()
    train_fail_num_subdata    = pd.read_csv(config.file_path_train_fail_annotation)["num_subdata"].tolist()
    eval_success_annots       = pd.read_csv(config.file_path_eval_success_annotation)["filename"].tolist()
    eval_success_num_subdata  = pd.read_csv(config.file_path_eval_success_annotation)["num_subdata"].tolist()
    eval_fail_annots          = pd.read_csv(config.file_path_eval_fail_annotation)["filename"].tolist()
    eval_fail_num_subdata     = pd.read_csv(config.file_path_eval_fail_annotation)["num_subdata"].tolist()
    
    # Dataset
    ranking_train_dataset = FetchingPreferenceValueonlyDataset(
        config = config.dataset_config,
        success_data_path_json = config.data_path_train_dataset_json,
        success_data_path_npz  = config.data_path_train_dataset_npz,
        success_annotations    = train_success_annots,
        success_num_subdata    = train_success_num_subdata,
        comparison_data_path_json    = config.data_path_train_dataset_json,
        comparison_data_path_npz     = config.data_path_train_dataset_npz,
        comparison_annotations       = train_fail_annots,
        comparison_num_subdata       = train_fail_num_subdata)
    ranking_eval_dataset = FetchingPreferenceValueonlyDataset(
        config = config.dataset_config,
        success_data_path_json = config.data_path_eval_dataset_json,
        success_data_path_npz  = config.data_path_eval_dataset_npz,
        success_annotations    = eval_success_annots,
        success_num_subdata    = eval_success_num_subdata,
        comparison_data_path_json    = config.data_path_eval_dataset_json,
        comparison_data_path_npz     = config.data_path_eval_dataset_npz,
        comparison_annotations       = eval_fail_annots,
        comparison_num_subdata       = eval_fail_num_subdata)


    # Sampler
    train_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = ranking_train_dataset.weights, 
            num_samples = len(ranking_train_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)
    eval_sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = ranking_eval_dataset.weights, 
            num_samples = len(ranking_eval_dataset), 
            replacement = True),
        batch_size = config.batch_size,
        drop_last = True)
    

    # Dataloader
    train_loader = DataLoader(
        ranking_train_dataset,
        batch_sampler = train_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)
    eval_loader = DataLoader(
        ranking_eval_dataset,
        batch_sampler = eval_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)


    # Model
    model = HistoryPlaceValueonly( 
        fetching_gpt_config = config.gpt_config,
        value_config        = config.value_config ).to(config.device)
    optimizer = Optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = Optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000, 4500], gamma=0.1)
    print(f"params: {get_n_params(model)}")


    # Metric
    loss_fn = nn.BCELoss()


    # Logger
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)
    logger = SummaryWriter(model_dir)
    # Save configuration
    config.save(model_dir + "/config.yaml")


    # Epoch loop
    start_epoch = 1
    steps = 1
    best_error = 10000.
    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')

        # Training        
        steps = train_epoch_fn(
            config       = config,
            epoch        = epoch,
            start_steps  = steps,
            train_loader = train_loader,
            model        = model,
            loss_fn      = loss_fn,
            optimizer    = optimizer,
            scheduler    = scheduler,
            logger       = logger)
        # Evaluating
        if epoch % config.eval_freq == 0:
            best_error = eval_epoch_fn(
                config      = config,
                epoch       = epoch,
                eval_loader = eval_loader,
                model       = model,
                loss_fn     = loss_fn,
                optimizer   = optimizer,
                scheduler   = scheduler,
                logger      = logger,
                best_error  = best_error,
                model_dir   = model_dir)
        
        print(f'===== End {epoch} epoch =====')




def train_epoch_fn(config: Setting,
                   epoch: int,
                   start_steps: int,
                   train_loader: DataLoader,
                   model: nn.Module,
                   loss_fn: nn.Module,
                   optimizer: Optim.Optimizer,
                   scheduler: Optim.lr_scheduler._LRScheduler,
                   logger: SummaryWriter) -> int:
    """1 epoch train function

    Args:
        config (Setting): Config file
        epoch (int): Current epoch
        start_steps (int): Steps before epoch
        train_loader (DataLoader): Data loader
        model (nn.Module): Model
        loss_fn (nn.Module): Loss fn
        optimizer (Optim.Optimizer): Optimizer
        scheduler (Optim.lr_scheduler._LRScheduler): Scheduler
        logger (SummaryWriter): Logger
    
    Returns:
        end_steps (int): steps after epoch
    """

    # Training one epoch
    batch_time       = AverageMeter('Time', ':6.3f')
    train_loss_meter = AverageMeter("Value total", ':.4e')
    progress = ProgressMeter(num_batches = len(train_loader),
                             meters = [batch_time, train_loss_meter],
                             prefix = "Epoch: [{}]".format(epoch))

    print("Training...")
    steps = start_steps
    model.train()
    for i, data in enumerate(train_loader):
        # NOTE(dr): ranking dataset outputs total 25 data. 
        #   Please check fetching_valueonly_dataset.py line 283~307
        #   It would be better to have if statement here to deal with ranking loss training
        
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
            fail_init_observation_rgbd,
            fail_init_observation_grasp,
            fail_goal,
            fail_seq_action,
            fail_seq_observation_rgbd,
            fail_seq_observation_grasp,
            fail_action_mask,
            fail_observation_mask,
            fail_input_trajectory_length,
            fail_full_trajectory_length,
            fail_future_discounted_reward_label,
            fail_success_or_fail,
            ranking) = ([d.to(config.device) for d in data])

        # Forward shape=(B)
        pred_success_ranking = model(
            success_init_observation_rgbd, success_init_observation_grasp, success_goal, 
            success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp,
            success_action_mask, success_observation_mask)
        pred_fail_ranking = model(
            fail_init_observation_rgbd, fail_init_observation_grasp, fail_goal, 
            fail_seq_action, fail_seq_observation_rgbd, fail_seq_observation_grasp,
            fail_action_mask, fail_observation_mask)
        pred_concat = torch.cat((pred_success_ranking, pred_fail_ranking), dim=-1)
        pred_concat = torch.softmax(pred_concat, dim=-1)
                
        # Loss calculation
        loss_total: torch.Tensor = loss_fn(pred_concat, ranking)

        # Backprop + Optimize ...
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # Logging (per step)
        batch_size = config.batch_size
        train_loss_meter.update(loss_total.item(), batch_size)
        logger.add_scalar('Loss(Ranking)/train_steps', loss_total.item(), steps)
        steps += 1
        if i % config.print_freq == 0:
            progress.display(i)

    # Logging
    logger.add_scalar('Loss(Ranking)/train', train_loss_meter.avg, epoch)

    return steps



def eval_epoch_fn(config: Setting,
                  epoch: int,
                  eval_loader: DataLoader,
                  model: nn.Module,
                  loss_fn: nn.Module,
                  optimizer: Optim.Optimizer,
                  scheduler: Optim.lr_scheduler._LRScheduler,
                  logger: SummaryWriter,
                  best_error: float,
                  model_dir: str) -> float:
    """1 epoch eval function

    Args:
        config (Setting): Config file
        epoch (int): Current epoch
        eval_loader (DataLoader): Data loader
        model (nn.Module): Model
        loss_fn (nn.Module): Loss fn
        optimizer (Optim.Optimizer): Optimizer
        scheduler (Optim.lr_scheduler._LRScheduler): Scheduler
        logger (SummaryWriter): Logger
        best_error (float): Best error
        model_dir (str): Checkpoint save dir
    
    Returns:
        best_error
    """
    if str(config.device) == 'cuda':
        torch.cuda.empty_cache()

    # Evaluate
    batch_time      = AverageMeter('Time', ':6.3f')
    eval_loss_meter = AverageMeter('Value total', ':.4e')
    progress = ProgressMeter(num_batches = len(eval_loader),
                             meters      = [batch_time, eval_loss_meter],
                             prefix      = "Epoch: [{}]".format(epoch))
    
    print("Validating...")
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            
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
                fail_init_observation_rgbd,
                fail_init_observation_grasp,
                fail_goal,
                fail_seq_action,
                fail_seq_observation_rgbd,
                fail_seq_observation_grasp,
                fail_action_mask,
                fail_observation_mask,
                fail_input_trajectory_length,
                fail_full_trajectory_length,
                fail_future_discounted_reward_label,
                fail_success_or_fail,
                ranking) = ([d.to(config.device) for d in data])
            
            # Forward shape=(B)
            pred_success_ranking = model(
                success_init_observation_rgbd, success_init_observation_grasp, success_goal, 
                success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp,
                success_action_mask, success_observation_mask)
            pred_fail_ranking = model(
                fail_init_observation_rgbd, fail_init_observation_grasp, fail_goal, 
                fail_seq_action, fail_seq_observation_rgbd, fail_seq_observation_grasp,
                fail_action_mask, fail_observation_mask)
            pred_concat = torch.cat((pred_success_ranking, pred_fail_ranking), dim=-1)
            pred_concat = torch.softmax(pred_concat, dim=-1)
            
            # Loss calculation
            loss_total: torch.Tensor = loss_fn(pred_concat, ranking)
            
            # Logging (per step)
            batch_size = config.batch_size
            eval_loss_meter.update(loss_total.item(), batch_size)
            if i % config.print_freq == 0:
                progress.display(i)

    # Logging
    logger.add_scalar('Loss(Ranking)/eval', eval_loss_meter.avg, epoch)

    # Save the last model
    save_checkpoint("Saving the last model!",
                    os.path.join(model_dir, "last.pth"),
                    epoch, eval_loss_meter.avg, 
                    model, optimizer, scheduler)

    # Save the best model
    if eval_loss_meter.avg < best_error:
        best_error = eval_loss_meter.avg
        save_checkpoint("Saving the best model!",
                        os.path.join(model_dir, "best.pth"),
                        epoch, best_error, 
                        model, optimizer, scheduler)
        
    return best_error



if __name__=="__main__":
    main(Setting())