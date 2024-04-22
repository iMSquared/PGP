import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter


import numpy as np
from tqdm import tqdm
import os
from simple_parsing import Serializable
from dataclasses import dataclass
from typing import Tuple, Dict
import pandas as pd
import cv2

from learning.dataset.fetching_image_dataset import FetchingImageDataset, normalize, denormaize
from learning.model.common.mae import MaskedAutoencoderViT
from learning.utils import save_checkpoint, AverageMeter, ProgressMeter, get_n_params


@dataclass
class Setting(Serializable):

    # Dataset
    dataname: str = "April17th"
    sim_or_exec: str = "sim_dataset"
    file_path_train_entire_annotation: str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/entire.csv"
    data_path_train_dataset_json     : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
    data_path_train_dataset_npz      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
    file_path_eval_entire_annotation : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/entire.csv"
    data_path_eval_dataset_json      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_eval_dataset_npz       : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"


    mae_config: MaskedAutoencoderViT.Config = MaskedAutoencoderViT.Config(
        img_size = 64,
        patch_size = 8,
        in_chans = 4,
        embed_dim = 128,
        depth = 4,
        num_heads = 8,
        decoder_embed_dim = 128,
        decoder_depth = 3,
        decoder_num_heads = 4,
        mlp_ratio = 4.,
        mask_ratio = 0.75,
        early_cnn = True,
        pred_reward = False
    )

    # Dataset params
    dataset_config: FetchingImageDataset.Config = FetchingImageDataset.Config(
            seq_length    = 6,
            image_res     = 64)
    
    # Training
    device       : str   = "cuda:0"
    num_workers  : int   = 8
    max_steps    : int   = 200000
    batch_size   : int   = 128
    learning_rate: float = 0.0001

    # Logging 
    exp_dir       : str = '/home/jiyong/workspace/POMDP/Simulation/pybullet_env/learning/exp'
    # model_name    : str = 'test'
    model_name    : str = f'4.20_value_mae_{dataname}_{sim_or_exec}_batch{batch_size}_lr{learning_rate}_cnn{mae_config.early_cnn}_emb{mae_config.embed_dim}_p{mae_config.patch_size}_m{mae_config.mask_ratio}'
    train_log_freq: int = 10
    eval_log_freq : int = 10 
    eval_save_freq: int = 100    # per training step



def main(config: Setting):

    # Train/eval split
    train_entire_annots      = pd.read_csv(config.file_path_train_entire_annotation)["filename"].tolist()
    train_entire_num_subdata = pd.read_csv(config.file_path_train_entire_annotation)["num_subdata"].tolist()
    eval_entire_annots       = pd.read_csv(config.file_path_eval_entire_annotation)["filename"].tolist()
    eval_entire_num_subdata  = pd.read_csv(config.file_path_eval_entire_annotation)["num_subdata"].tolist()
    
    # Dataset
    train_dataset = FetchingImageDataset(
        config = config.dataset_config,
        data_path_json    = config.data_path_train_dataset_json,
        data_path_npz     = config.data_path_train_dataset_npz,
        annotations       = train_entire_annots,
        num_subdata       = train_entire_num_subdata)
    eval_dataset = FetchingImageDataset(
        config = config.dataset_config,
        data_path_json    = config.data_path_eval_dataset_json,
        data_path_npz     = config.data_path_eval_dataset_npz,
        annotations       = eval_entire_annots,
        num_subdata       = eval_entire_num_subdata)

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
    model = MaskedAutoencoderViT(config.mae_config).to(config.device)
    optimizer = Optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = Optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000, 4500], gamma=0.1)
    print(f"params: {get_n_params(model)}")

    # Metric
    loss_fn = model.forward_loss

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
        loss_fn   = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger    = logger)
    eval_wrapper = EvalWrapper(
        config    = config,
        loader    = eval_loader,
        model     = model,
        loss_fn   = loss_fn,
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
        imgs = data.to(self.config.device)
        
        # Normalize
        imgs, min_depth, max_depth = normalize(imgs)

        # Forward & loss calculation
        loss, pred, mask = self.model(imgs, mask_ratio=self.config.mae_config.mask_ratio)
        
        # Backprop + Optimize ...
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            # Denormalize
            pred = self.model.unpatchify(pred)
            pred = denormaize(pred, min_depth, max_depth)
            
            # Masked input
            patchwise_imgs = self.model.patchify(imgs)
            N, L, C = patchwise_imgs.shape
            expanded_mask = mask.unsqueeze(-1).expand(N, L, C)
            masked_patchwise_imgs = patchwise_imgs * (1 - expanded_mask)
            masked_imgs = self.model.unpatchify(masked_patchwise_imgs)

        # Logging 1 (per steps)
        self.logger.add_scalar('Loss(MSE)/train_steps', loss.item(), steps)
        # Logging 2 (console)
        self.loss_value_meter.update(loss.item())
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
            imgs = data.to(self.config.device)
            
            # Normalize
            imgs, min_depth, max_depth = normalize(imgs)

            # Forward & loss calculation
            loss, pred, mask = self.model(imgs, mask_ratio=self.config.mae_config.mask_ratio)

            # Denormalize
            pred = self.model.unpatchify(pred)
            pred = denormaize(pred, min_depth, max_depth)
            
            # Masked input
            patchwise_imgs = self.model.patchify(imgs)
            N, L, C = patchwise_imgs.shape
            expanded_mask = mask.unsqueeze(-1).expand(N, L, C)
            masked_patchwise_imgs = patchwise_imgs * (1 - expanded_mask)
            masked_imgs = self.model.unpatchify(masked_patchwise_imgs)

        # Logging 1 (tensorboard)
        self.logger.add_scalar('Loss(MSE)/eval', loss.item(), steps)
        # Logging 2 (console)
        self.loss_value_meter.update(loss.item())
        print(f"[Eval  {steps}] {self.loss_value_meter}")
        self.__reset_meters()

        # Save the last model
        if steps % self.config.eval_save_freq == 0:
            save_checkpoint("Saving the last model!",
                            os.path.join(self.model_dir, "last.pth"),
                            -1, loss.item(), 
                            self.model, self.optimizer, self.scheduler)
            
            # Image tensors to numpy
            imgs = denormaize(imgs, min_depth, max_depth)
            img = imgs[0].detach().cpu().numpy()
            masked_img = masked_imgs[0].detach().cpu().numpy()
            pred_img = pred[0].detach().cpu().numpy()
            
            # Save images
            rgb_img = img[1:4, :, :].astype(np.uint8)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            masked_rgb_img = masked_img[1:4, :, :].astype(np.uint8)
            masked_rgb_img = np.transpose(masked_rgb_img, (1, 2, 0))
            pred_rgb_img = pred_img[1:4, :, :].astype(np.uint8)
            pred_rgb_img = np.transpose(pred_rgb_img, (1, 2, 0))
            
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_input_RGB.png"), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_input_RGB_masked.png"), cv2.cvtColor(masked_rgb_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_output_RGB.png"), cv2.cvtColor(pred_rgb_img, cv2.COLOR_RGB2BGR))
            
            depth_img = img[0, :, :]
            depth_img = 255. * (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))
            depth_img = depth_img.astype(np.uint8)

            masked_depth_img = masked_img[0, :, :]
            masked_depth_img = 255. * (masked_depth_img - np.min(masked_depth_img)) / (np.max(masked_depth_img) - np.min(masked_depth_img))
            masked_depth_img = masked_depth_img.astype(np.uint8)
            
            pred_depth_img = pred_img[0, :, :]
            pred_depth_img = 255. * (pred_depth_img - np.min(pred_depth_img)) / (np.max(pred_depth_img) - np.min(pred_depth_img))
            pred_depth_img = pred_depth_img.astype(np.uint8)
            
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_input_depth.png"), depth_img)
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_input_depth_masked.png"), masked_depth_img)
            cv2.imwrite(os.path.join(self.model_dir, f"{steps}_output_depth.png"), pred_depth_img)
            

        # Save the best model
        if loss.item() < self.best_error:
            self.best_error = loss.item()
            save_checkpoint("Saving the best model!",
                            os.path.join(self.model_dir, "best.pth"),
                            -1, self.best_error, 
                            self.model, self.optimizer, self.scheduler)



if __name__=="__main__":
    main(Setting())