import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
from torch.utils.data import DataLoader

import os
from simple_parsing import Serializable
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from dataset.placepolicy_valueonly_dataset import PlacePolicyValueonlyDataset, PlacePolicyValueonlyDatasetCollateFn
from model.policy.place_gpt import HistoryPlaceValueonly, GPT2FetchingPlaceConditioner
from model.value.value import ValueNet
from utils import load_checkpoint_inference

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Setting(Serializable):

    # Dataset
    data_path_positive_json: str = "/home/sanghyeon/vessl/cql/sim_positive_json"
    data_path_positive_npz : str = "/home/sanghyeon/vessl/cql/sim_positive_npz"
    data_path_negative_json: str = "/home/sanghyeon/vessl/cql/sim_negative_json"
    data_path_negative_npz : str = "/home/sanghyeon/vessl/cql/sim_negative_npz"


    # This parameter governs the model size
    dim_model_hidden = 128

    dim_gpt_hidden = dim_model_hidden
    dim_condition  = dim_model_hidden
    gpt_config: GPT2FetchingPlaceConditioner.Config = GPT2FetchingPlaceConditioner.Config(
        # Data type
        image_res        = 64,
        dim_obs_ch       = 4,    # RGBD
        dim_encode_obs   = 128,
        dim_action_input = 11,   
        dim_reward_input = 1,
        dim_goal_input   = 5,
        # Architecture
        dim_hidden       = dim_gpt_hidden,
        num_heads        = 2,
        dim_ffn          = 256,
        num_gpt_layers   = 3,
        dropout_rate     = 0.1,
        # Positional encoding
        max_len          = 100,
        seq_len          = 3,
        # Output
        dim_condition    = dim_condition)
    
    value_config: ValueNet.Config = ValueNet.Config(
        dim_condition = dim_model_hidden)


    # Training
    device         : str   = "cuda:0"
    resume         : bool  = False
    pre_trained    : bool  = False
    
    value_loss_beta: float = 0.0001

    num_workers    : int   = 4
    epochs         : int   = 5000
    batch_size     : int   = 1024
    learning_rate  : float = 0.001

    # Logging 
    exp_dir        : str   = '/home/sanghyeon/workspace/POMDP_multiobject/Simulation/pybullet_env/learning/exp'
    exp            : str   = f'3.27_randominit_valueonly_lr{learning_rate}_beta{value_loss_beta}'
    model_name     : str   = "best.pth"
    value_plot_dir : str   = '/home/sanghyeon/workspace/POMDP_multiobject/Simulation/pybullet_env/learning/value_plot_dir'




def main(config: Setting):

    # Train/eval split
    annots = sorted(os.listdir(config.data_path_positive_json))
    train_positive_annot, eval_positive_annot = train_test_split(annots, train_size=0.9, random_state=42)
    annots_negative = sorted(os.listdir(config.data_path_negative_json))
    train_negative_annot, eval_negative_annot = train_test_split(annots_negative, train_size=0.9, random_state=42)
    
    # Dataset
    train_positive_dataset = PlacePolicyValueonlyDataset(
        config             = PlacePolicyValueonlyDataset.Config(),
        data_path_json     = config.data_path_positive_json,
        data_path_npz      = config.data_path_positive_npz,
        annotations_json   = train_positive_annot)
    eval_positive_dataset = PlacePolicyValueonlyDataset(
        config             = PlacePolicyValueonlyDataset.Config(),
        data_path_json     = config.data_path_positive_json,
        data_path_npz      = config.data_path_positive_npz,
        annotations_json   = eval_positive_annot)    

    train_negative_dataset = PlacePolicyValueonlyDataset(
        config             = PlacePolicyValueonlyDataset.Config(),
        data_path_json     = config.data_path_negative_json,
        data_path_npz      = config.data_path_negative_npz,
        annotations_json   = train_negative_annot)    
    eval_negative_dataset = PlacePolicyValueonlyDataset(
        config             = PlacePolicyValueonlyDataset.Config(),
        data_path_json     = config.data_path_negative_json,
        data_path_npz      = config.data_path_negative_npz,
        annotations_json   = eval_negative_annot)
    
    # Dataloader
    train_positive_loader = DataLoader(
        train_positive_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        drop_last   = True,
        pin_memory  = True,
        collate_fn  = PlacePolicyValueonlyDatasetCollateFn())
    eval_positive_loader = DataLoader(
        eval_positive_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        drop_last   = True,
        pin_memory  = True,
        collate_fn  = PlacePolicyValueonlyDatasetCollateFn())
    
    train_negative_loader = DataLoader(
        train_negative_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        drop_last   = True,
        pin_memory  = True,
        collate_fn  = PlacePolicyValueonlyDatasetCollateFn())
    eval_negative_loader = DataLoader(
        eval_negative_dataset,
        batch_size  = config.batch_size,
        shuffle     = True,
        num_workers = config.num_workers,
        drop_last   = True,                 # Make sure to set it true,, otherwise it will crash
        pin_memory  = True,
        collate_fn  = PlacePolicyValueonlyDatasetCollateFn())

    # Model
    model = HistoryPlaceValueonly( fetching_gpt_config = config.gpt_config,
                                   value_config = config.value_config ).to(config.device)
    model_dir = os.path.join(config.exp_dir, config.exp)
    model_name = config.model_name
    filename = os.path.join(model_dir, model_name)
    load_checkpoint_inference(config.device, filename, model)

    # Plot logger
    plot_dir = os.path.join(config.value_plot_dir, config.exp)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # Save configuration
    config.save(model_dir + "/config.yaml")


    # Evalutation
    negative_iter = iter(train_negative_loader)
    negative_iter_eval = iter(eval_negative_loader)

    train_pred_positive_value_list = []
    train_pred_negative_value_list = []
    train_gt_positive_value_list = []
    train_gt_negative_value_list = []
    # Evaluation on training set
    print("Evaluating on training set...")
    model.eval()
    with torch.no_grad():
        for i, positive_data in enumerate(tqdm(train_positive_loader)):
            try:
                negative_data = next(negative_iter)
            except StopIteration:
                print("Restarting negative training loader")
                negative_iter = iter(train_negative_loader)
                negative_data = next(negative_iter)

            # To device...
            init_obs_positive, goal_positive, \
                seq_action_positive, seq_obs_positive, seq_reward_positive, \
                mask_positive, time_step_to_predict_postive, \
                future_discounted_reward_positive = ([d.to(config.device) for d in positive_data])
            
            init_obs_negative, goal_negative, \
                seq_action_negative, seq_obs_negative, seq_reward_negative, \
                mask_negative, time_step_to_predict_negative, \
                future_discounted_reward_negative = ([d.to(config.device) for d in negative_data])

            # Forward            
            pred_positive_value = model(init_obs_positive, goal_positive,
                                        seq_action_positive, seq_obs_positive, seq_reward_positive,
                                        mask_positive)

            pred_negative_value = model(init_obs_negative, goal_negative,
                                        seq_action_negative, seq_obs_negative, seq_reward_negative,
                                        mask_negative)

            # Ploting
            train_pred_positive_value_list += pred_positive_value.squeeze().cpu().numpy().tolist()
            train_pred_negative_value_list += pred_negative_value.squeeze().cpu().numpy().tolist()
            train_gt_positive_value_list += future_discounted_reward_positive.squeeze().cpu().numpy().tolist()
            train_gt_negative_value_list += future_discounted_reward_negative.squeeze().cpu().numpy().tolist()


    eval_pred_positive_value_list = []
    eval_pred_negative_value_list = []
    eval_gt_positive_value_list = []
    eval_gt_negative_value_list = []
    # Evaluation on validation set
    if str(config.device) == 'cuda':
        torch.cuda.empty_cache()
    print("Evaluating on validation set...")
    model.eval()
    with torch.no_grad():
        for i, positive_data in enumerate(tqdm(eval_positive_loader)):
            try:
                negative_data = next(negative_iter_eval)
            except StopIteration:
                print("Restarting negative eval loader")
                negative_iter_eval = iter(eval_negative_loader)
                negative_data = next(negative_iter_eval)

            # To device...                    
            init_obs_positive, goal_positive, \
                seq_action_positive, seq_obs_positive, seq_reward_positive, \
                mask_positive, time_step_to_predict_positive, \
                future_discounted_reward_positive = ([d.to(config.device) for d in positive_data])
            
            init_obs_negative, goal_negative, \
                seq_action_negative, seq_obs_negative, seq_reward_negative, \
                mask_negative, time_step_to_predict_negative, \
                future_discounted_reward_negative = ([d.to(config.device) for d in negative_data])
            

            # Forward
            pred_positive_value = model(init_obs_positive, goal_positive,
                                        seq_action_positive, seq_obs_positive, seq_reward_positive,
                                        mask_positive)
    
            pred_negative_value = model(init_obs_negative, goal_negative,
                                        seq_action_negative, seq_obs_negative, seq_reward_negative,
                                        mask_negative)
            
            # Ploting
            eval_pred_positive_value_list += pred_positive_value.squeeze().cpu().numpy().tolist()
            eval_pred_negative_value_list += pred_negative_value.squeeze().cpu().numpy().tolist()
            eval_gt_positive_value_list += future_discounted_reward_positive.squeeze().cpu().numpy().tolist()
            eval_gt_negative_value_list += future_discounted_reward_negative.squeeze().cpu().numpy().tolist()


    # Drawing plot...
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = (12, 10)
    plt.rcParams["font.size"] = 8

    fig, axes = plt.subplots(2, 2)
    
    axes[0][0].boxplot([train_pred_positive_value_list, train_pred_negative_value_list], 
                       vert = False,
                       labels = ["Positive", "Negative"])
    axes[0][0].set_xlim(-110, 110)
    axes[0][0].set_xlabel("Value")
    axes[0][0].set_ylabel("Type")
    axes[0][0].set_title("Prediction (Train)")
    
    axes[1][0].boxplot([train_gt_positive_value_list, train_gt_negative_value_list], 
                       vert = False,
                       labels = ["Positive", "Negative"])
    axes[1][0].set_xlim(-110, 110)
    axes[1][0].set_xlabel("Value")
    axes[1][0].set_ylabel("Type")
    axes[1][0].set_title("Groundtruth (Train)")

    axes[0][1].boxplot([eval_pred_positive_value_list, eval_pred_negative_value_list], 
                       vert = False,
                       labels = ["Positive", "Negative"])
    axes[0][1].set_xlim(-110, 110)
    axes[0][1].set_xlabel("Value")
    axes[0][1].set_ylabel("Type")
    axes[0][1].set_title("Prediction (Validation)")

    axes[1][1].boxplot([eval_gt_positive_value_list, eval_gt_negative_value_list], 
                       vert = False,
                       labels = ["Positive", "Negative"])
    axes[1][1].set_xlim(-110, 110)
    axes[1][1].set_xlabel("Value")
    axes[1][1].set_ylabel("Type")
    axes[1][1].set_title("Groundtruth (Validation)")

    plt.show()


if __name__=="__main__":
    main(Setting())