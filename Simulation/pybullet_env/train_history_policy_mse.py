import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
# import wandb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import numpy as np
from tqdm import tqdm
import os
from simple_parsing import Serializable
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import yaml

from learning.dataset.fetching_policy_dataset import FetchingPolicyDataset
from learning.dataset.common import FetchingDatasetConfig
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.common.cvae import CVAE
from learning.model.policy import HistoryPlacePolicyonly
from learning.utils import save_checkpoint, get_n_params, AverageMeter
from learning.loss import ELBOLoss


@dataclass
class Setting(Serializable):

    # Dataset
    dataname: str = "May6th_3obj_depth8_3000"
    sim_or_exec: str = "sim_dataset"
    file_path_train_success_annotation: str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/success.csv"
    data_path_train_dataset_json      : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
    data_path_train_dataset_npz       : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
    file_path_eval_success_annotation : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/success.csv"
    data_path_eval_dataset_json       : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_eval_dataset_npz        : str = f"/home/shared_directory/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"

    # This parameter governs the model size
    dim_model_hidden: int = 256
    seq_length      : int = 8
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
    # CVAE head params
    dim_action_output: int = 3
    dim_cvae_embed   : int = 64
    dim_vae_latent   : int = dim_model_hidden
    cvae_config: CVAE.Config \
        = CVAE.Config(
            latent_size         = dim_vae_latent,
            dim_condition       = dim_condition,
            dim_output          = dim_action_output,  # Action output
            dim_embed           = dim_cvae_embed,
            encoder_layer_sizes = (dim_cvae_embed, dim_cvae_embed + dim_condition, dim_vae_latent, dim_vae_latent),
            decoder_layer_sizes = (dim_vae_latent, dim_vae_latent + dim_condition, dim_vae_latent, dim_action_output))

    # Dataset params
    dataset_config: FetchingDatasetConfig \
        = FetchingDatasetConfig(
            seq_length = seq_length,
            image_res  = 64)

    # Training
    device       : str        = "cuda:1"
    num_workers  : int        = 32
    max_steps    : int        = 2000000
    batch_size   : int        = 64
    learning_rate: float      = 0.0001
    milestones   : Tuple[int] = (40000, 150000)
    beta         : float      = 0.25

    # Logging 
    exp_dir         : str = '/home/sanghyeon/workspace/POMDP/Simulation/pybullet_env/learning/exp'
    # model_name      : str = 'test'
    model_name      : str = f'5.06_policy_mse_{dataname}_{sim_or_exec}_dim{dim_model_hidden}_beta{beta}_batch{batch_size}_lr{learning_rate}'
    train_log_freq  : int = 10
    eval_num_samples: int = 1024                                # Minimum number of samples to use for evaluation metric.
    eval_log_freq   : int = int(eval_num_samples*train_log_freq*2/batch_size)
    eval_save_freq  : int = eval_log_freq         # per training step


def main(config: Setting):

    # wandb.init(
    #     project = "history_policy",
    #     config={
    #         "learning_rate": config.learning_rate,
    #         "batch_size": config.batch_size,
    #         "policy": True,
    #         "value": False,
    #         "vae_beta": config.vae_beta},
    #     sync_tensorboard=True )


    # Train/eval split
    train_success_annots      = pd.read_csv(config.file_path_train_success_annotation)["filename"].tolist()
    train_success_num_subdata = pd.read_csv(config.file_path_train_success_annotation)["num_subdata"].tolist()
    eval_success_annots       = pd.read_csv(config.file_path_eval_success_annotation)["filename"].tolist()
    eval_success_num_subdata  = pd.read_csv(config.file_path_eval_success_annotation)["num_subdata"].tolist()

    # Open pybullet config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", "config_primitive_object.yaml"), "r") as f:
        sim_config = yaml.load(f, Loader=yaml.FullLoader)

    # Dataset
    train_dataset = FetchingPolicyDataset(
        config         = config.dataset_config,
        data_path_json = config.data_path_train_dataset_json,
        data_path_npz  = config.data_path_train_dataset_npz,
        filenames      = train_success_annots,
        num_subdata    = train_success_num_subdata)
    eval_dataset = FetchingPolicyDataset(
        config         = config.dataset_config,
        data_path_json = config.data_path_eval_dataset_json,
        data_path_npz  = config.data_path_eval_dataset_npz,
        filenames      = eval_success_annots,
        num_subdata    = eval_success_num_subdata)
    
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
    train_policy_loader = DataLoader(
        train_dataset,
        batch_sampler = train_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)
    eval_policy_loader = DataLoader(
        eval_dataset,
        batch_sampler = eval_sampler,
        num_workers   = config.num_workers,
        pin_memory    = True)
    
    # Model
    model = HistoryPlacePolicyonly( 
        cvae_config         = config.cvae_config,
        fetching_gpt_config = config.gpt_config   ).to(config.device)
    optimizer = Optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = Optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    print(f"params: {get_n_params(model)}")

    # Metric
    cvae_loss_fn = ELBOLoss(config.beta)

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
        loader    = train_policy_loader,
        model     = model,
        loss_fn   = cvae_loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger)
    eval_wrapper = EvalWrapper(
        config    = config,
        loader    = eval_policy_loader,
        model     = model,
        loss_fn   = cvae_loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger,
        model_dir = model_dir,
        sim_config = sim_config)
    

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
        self.loss_kld_meter   = AverageMeter('KL-divergence', ':.4e')
        self.loss_recon_meter = AverageMeter('Reconstruction Error', ':4e')
        self.loss_elbo_meter  = AverageMeter('ELBO', ':4e')


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
            next_action_label, next_action_token = ([d.to(self.config.device) for d in data])

        # Forward
        recon_x, mean, log_var = self.model(next_action_label,
                                            init_obs_rgbd, init_obs_grasp, goal,
                                            seq_action, seq_obs_rgbd, seq_obs_grasp,
                                            action_mask, obs_mask)

        # Loss calculation
        cvae_loss: ELBOLoss.LossData = self.loss_fn(recon_x, next_action_label, mean, log_var)
        total_loss = cvae_loss.total

        # Backprop + Optimize ...
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Log loss
        self.loss_kld_meter.update(cvae_loss.kld.item())
        self.loss_recon_meter.update(cvae_loss.recon.item())
        self.loss_elbo_meter.update(cvae_loss.total.item())
        # Logging 1 (per steps)
        self.logger.add_scalar('Loss(CVAE_KL_divergence)/train_steps', cvae_loss.kld.item(), steps)
        self.logger.add_scalar('Loss(CVAE_Reconstruction)/train_steps', cvae_loss.recon.item(), steps)
        self.logger.add_scalar('Loss(CVAE_total)/train_steps', cvae_loss.total.item(), steps)
        # Logging 2 (console)
        if steps % self.config.train_log_freq == 0:
            print(f"[Train {steps}] {self.loss_elbo_meter}, {self.loss_kld_meter}, {self.loss_recon_meter}")
            self.__reset_meters()
        



class EvalWrapper:

    def __init__(self, config: Setting,
                       loader: DataLoader,
                       model: nn.Module,
                       loss_fn: nn.Module,
                       optimizer: Optim,
                       scheduler: Optim.lr_scheduler._LRScheduler,
                       logger: SummaryWriter,
                       model_dir: str,
                       sim_config):
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
        self.sim_config = sim_config
        self.__reset_meters()

    
    def __reset_meters(self):
        self.loss_kld_meter   = AverageMeter('KL-divergence', ':.4e')
        self.loss_recon_meter = AverageMeter('Reconstruction Error', ':4e')
        self.loss_elbo_meter  = AverageMeter('ELBO', ':4e')


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
                init_obs_rgbd, init_obs_grasp, goal, \
                    seq_action, seq_obs_rgbd, seq_obs_grasp, \
                    action_mask, obs_mask, \
                    input_trajectory_length, full_trajectory_length, \
                    next_action_label, next_action_token = ([d.to(self.config.device) for d in data])

                # Forward
                recon_x, mean, log_var = self.model(next_action_label,
                                                    init_obs_rgbd, init_obs_grasp, goal,
                                                    seq_action, seq_obs_rgbd, seq_obs_grasp,
                                                    action_mask, obs_mask)

                # Loss calculation
                cvae_loss: ELBOLoss.LossData = self.loss_fn(recon_x, next_action_label, mean, log_var)

                # Log loss
                self.loss_kld_meter.update(cvae_loss.kld.item())
                self.loss_recon_meter.update(cvae_loss.recon.item())
                self.loss_elbo_meter.update(cvae_loss.total.item())

                # Repeat evaluation...
                num_samples_total += self.config.batch_size


        # Logging 1 (tensorboard)
        self.logger.add_scalar('Loss(CVAE_KL_divergence)/eval', self.loss_kld_meter.avg , steps)
        self.logger.add_scalar('Loss(CVAE_Reconstruction)/eval', self.loss_recon_meter.avg, steps)
        self.logger.add_scalar('Loss(CVAE_total)/eval', self.loss_elbo_meter.avg, steps)
        # Logging 2 (console)
        print(f"[Eval  {steps}] {self.loss_elbo_meter}, {self.loss_kld_meter}, {self.loss_recon_meter}")
        last_avg_error = self.loss_elbo_meter.avg
        self.__reset_meters()


        # Save the last model
        if steps % self.config.eval_save_freq == 0:            
            self.eval_visualization(   
                steps,     
                init_obs_rgbd, init_obs_grasp, goal,
                seq_action, seq_obs_rgbd, seq_obs_grasp,
                action_mask, obs_mask,
                input_trajectory_length, full_trajectory_length)
            save_checkpoint("Saving the last model!",
                            os.path.join(self.model_dir, f"checkpoint{steps}.pth"),
                            -1, last_avg_error, 
                            self.model, self.optimizer, self.scheduler)

        # Save the best model
        if last_avg_error < self.best_error:
            self.best_error = last_avg_error
            save_checkpoint("Saving the best model!",
                            os.path.join(self.model_dir, "best.pth"),
                            -1, self.best_error, 
                            self.model, self.optimizer, self.scheduler)


    def eval_visualization(self, steps,
                                 init_obs_rgbd, init_obs_grasp, goal, \
                                 seq_action, seq_obs_rgbd, seq_obs_grasp, \
                                 action_mask, obs_mask, \
                                 input_trajectory_length, full_trajectory_length,):
        """Eval visualization
        
        NOTE(ssh): This code is terrible.
        """

        num_viz_per_traj = 64

        # Eval?
        self.model.eval()
        with torch.no_grad():
            
            fig, axes = plt.subplots(2, 4, figsize=(5*4, 5*2))
            axes = np.reshape(axes, (-1,))

            for i, _inputs in enumerate(zip(init_obs_rgbd, init_obs_grasp, goal, \
                                            seq_action, seq_obs_rgbd, seq_obs_grasp, \
                                            action_mask, obs_mask, input_trajectory_length)):
                # 8 plots...
                if i >= 8:
                    break

                # Repeat inference for same input
                _init_obs_rgbd, _init_obs_grasp, _goal, \
                    _seq_action, _seq_obs_rgbd, _seq_obs_grasp, \
                    _action_mask, _obs_mask, _input_trajectory_length = [
                        d.expand(num_viz_per_traj, *d.shape).contiguous()
                        for d in _inputs]
                # Inference
                recon_x = self.model.inference(
                    _init_obs_rgbd, _init_obs_grasp, _goal, \
                    _seq_action, _seq_obs_rgbd, _seq_obs_grasp, \
                    _action_mask, _obs_mask)
                
                
                is_target = torch.argmax(_seq_action[0,_input_trajectory_length[0].item()-1,2:4]).bool().item()

                # Visualize
                scatter = []
                colors = []
                for xyyaw in recon_x:
                    color = [1, 0, 0] if is_target else [0, 1, 0]
                    scatter.append(xyyaw.cpu().numpy())
                    colors.append(color)
                scatter = np.asarray(scatter)


                cabinet_pos = self.sim_config["env_params"]["binpick_env"]["cabinet"]["pos"]
                goal_pos = self.sim_config["env_params"]["binpick_env"]["goal"]["pos"]
                CABINET_POS_X = cabinet_pos[0]
                CABINET_POS_Y = cabinet_pos[1]
                CABINET_GRASP_MARGIN_X = 0.1
                CABINET_GRASP_MARGIN_Y = 0.2
                GOAL_POS_X = goal_pos[0]
                GOAL_POS_Y = goal_pos[1]
                min_x = CABINET_POS_X-CABINET_GRASP_MARGIN_X
                max_x = CABINET_POS_X+CABINET_GRASP_MARGIN_X
                min_y = CABINET_POS_Y-CABINET_GRASP_MARGIN_Y
                max_y = CABINET_POS_Y+CABINET_GRASP_MARGIN_Y

                axes[i].scatter(scatter[:,0], scatter[:,1], c=colors)
                axes[i].set_xlim(-0.2, 0.8)
                axes[i].set_ylim(-0.5, 0.5)
                goal_circle = patches.Circle((GOAL_POS_X, GOAL_POS_Y), radius=0.025, color=[0.7, 0.7, 0.7])
                col_1 = PatchCollection([goal_circle], zorder=0, match_original=True)
                axes[i].add_collection(col_1)
                axes[i].plot((min_x, min_x), (min_y, max_y), color='b')
                axes[i].plot((min_x, max_x), (max_y, max_y), color='b')
                axes[i].plot((max_x, max_x), (max_y, min_y), color='b')
                axes[i].plot((max_x, min_x), (min_y, min_y), color='b')

            model_dir = os.path.join(self.config.exp_dir, self.config.model_name)
            plt.savefig(os.path.join(model_dir, f"{steps}.png"))





if __name__=="__main__":
    main(Setting())