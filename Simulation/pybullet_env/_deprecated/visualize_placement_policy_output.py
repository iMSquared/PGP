import os
import glob
import pickle
import random

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from simple_parsing import Serializable
from dataclasses import dataclass, field

from learning.dataset.process_policy_data import PlacePolicySimulationBeliefDataset
from learning.model.policy.place import PolicyModelPlaceBelief
from learning.script.utils import load_checkpoint

    
    
if __name__ == '__main__':
    
    @dataclass
    class Settings(Serializable):
        # dataset
        train_data_path        : str   = '/home/sanghyeon/vessl/sim_dataset_fixed_belief_train'
        eval_data_path         : str   = '/home/sanghyeon/vessl/sim_dataset_fixed_belief_eval'
        # input
        dim_action_place       : int   = 3
        dim_action_embed       : int   = 16        
        dim_point              : int   = 7
        dim_goal               : int   = 5
        # PointNet
        num_point              : int   = 2048
        dim_pointnet           : int   = 128
        dim_goal_hidden        : int   = 8
        # CVAE
        dim_vae_latent         : int   = 16
        dim_vae_condition      : int   = 16
        vae_encoder_layer_sizes: Tuple = (dim_action_embed, dim_action_embed + dim_vae_condition, dim_vae_latent)
        vae_decoder_layer_sizes: Tuple = (dim_vae_latent, dim_vae_latent + dim_vae_condition, dim_action_place)
        # Training
        device                 : int   = 'cuda' if th.cuda.is_available() else 'cpu'
        # device                 : int   = 'cpu'
        resume                 : int   = 'best.pth' # checkpoint file name for resuming
        pre_trained            : bool  = None
        epochs                 : int   = 5000
        batch_size             : int   = 16
        learning_rate          : float = 1e-4
        # Logging
        exp_dir                : str   = '/home/ajy8456/workspace/POMDP/learning/exp'
        model_name             : str   = '2.26_place_policy_belief'
        print_freq             : int   = 10 # per training step
        train_eval_freq        : int   = 100 # per training step
        eval_freq              : int   = 10 # per epoch
        save_freq              : int   = 100 # per epoch
    
    config = Settings()
    num_output = 100
    
    dataset = glob.glob(f'{config.eval_data_path}/*.pickle')
    index = np.random.choice(len(dataset))
    
    while True:
    
        with open(dataset[index], 'rb') as f:
            traj = pickle.load(f)
        
        traj_b = traj['belief']
        traj_a = traj['action']
        
        place_steps = []
        for i, a in enumerate(traj_a):
            if a[0] == 'PLACE':
                place_steps.append(i)
        
        sampled_step = random.choice(place_steps)
        
        b = traj_b[sampled_step]
        num_point_in_b = len(b)
        p_weights = b[:,-1]
        p_weights /= np.sum(p_weights)
        b[:,-1] = p_weights

        if num_point_in_b >= config.num_point:
            p_indices = np.random.choice(num_point_in_b, config.num_point, replace=False, p = b[:,-1])
        else:
            p_indices = np.random.choice(num_point_in_b, config.num_point, replace=True, p = b[:,-1])
        
        print(traj_a[sampled_step-1])
        
        if traj_a[sampled_step-1][1] == 'X':
            break
    
    g = th.tensor(traj['goal'], dtype=th.float32).to(config.device)
    b = th.tensor(traj_b[sampled_step][p_indices,:], dtype=th.float32).to(config.device)
    a = th.tensor([traj_a[sampled_step][2][0], traj_a[sampled_step][2][1], traj_a[sampled_step][4]], dtype=th.float32).to(config.device)
    
    data = {
        'goal': g,
        'belief': b,
        'target_action': a
    }
    
    # Model
    model_dir = os.path.join(config.exp_dir, config.model_name)
    device    = config.device
    model     = PolicyModelPlaceBelief(config).to(device)
    model.eval()
    # Optimizer
    # |TODO(jiyong)|: Make it compatible with the config
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 4000, 4500])
    # Load checkpoint for resuming
    filename = os.path.join(model_dir, config.resume)
    start_epoch, model, optimizer, scheduler = load_checkpoint(config, 
                                                                                filename, 
                                                                                model, 
                                                                                optimizer, 
                                                                                scheduler)
    start_epoch += 1
    print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
    
    
    model.eval()
    p = data['belief'].to(config.device).expand(num_output, config.num_point, config.dim_point)
    c = data['goal'].to(config.device)
    pred = model.inference(p, c).squeeze()
    
    # Plot
    pcd = data['belief'][:, 0:3].to('cpu').numpy()
    output = pred.detach().to('cpu')
    output = output.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0.0, 1.0])

    ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c='blue')
    ax.scatter(output[:,0], output[:,1], 0.39, s=0.2, c='red')

    plt.savefig('test.png')