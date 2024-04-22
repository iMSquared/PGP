import os
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dataclasses import dataclass, replace
from simple_parsing import Serializable

import sys
sys.path.append('Simulation/pybullet_env/learning')
from dataset.process_policy_data import PlacePolicySimulationBeliefDataset
from model.policy.place import PolicyModelPlaceBelief
from script.utils import TorchLog, Trainer, Evaluator, save_checkpoint, load_checkpoint
from script.loss import ELBOLoss


@dataclass
class Settings(Serializable):
    # dataset
    train_data_path = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_train'
    eval_data_path = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_eval'
    
    # input
    dim_action_place = 3
    dim_action_embed = 16        
    dim_point = 7
    dim_goal = 5
    
    # PointNet
    num_point = 512
    dim_pointnet = 128
    dim_goal_hidden = 8

    # CVAE
    dim_vae_latent = 16
    dim_vae_condition = 16
    vae_encoder_layer_sizes = [dim_action_embed, dim_action_embed + dim_vae_condition, dim_vae_latent]
    vae_decoder_layer_sizes = [dim_vae_latent, dim_vae_latent + dim_vae_condition, dim_action_place]
    
    # Training
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    resume = None # checkpoint file name for resuming
    pre_trained  = None
    
    epochs = 5000
    batch_size = 16
    learning_rate = 1e-4
    
    # Logging
    exp_dir: str = '/home/ajy8456/workspace/POMDP/learning/exp'
    model_name: str = '2.25_place_poliy_belief_test'
    
    print_freq = 10 # per training step
    train_eval_freq = 100 # per training step
    eval_freq: int = 10 # per epoch
    save_freq: int = 100 # per epoch
    

def main(config):
    # Dataset
    train_dataset = PlacePolicySimulationBeliefDataset(config, config.train_data_path)
    eval_dataset = PlacePolicySimulationBeliefDataset(config, config.eval_data_path)
    train_data_loader = DataLoader(train_dataset, config.batch_size)
    eval_data_loader = DataLoader(eval_dataset, config.batch_size)

    # Model
    model = PolicyModelPlaceBelief(config).to(config.device)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 4000, 4500])

    # Logger
    model_dir = os.path.join(config.exp_dir, config.model_name)
    logger = SummaryWriter(model_dir)

    
    # save configuration
    config.save(model_dir + '/config.yaml')
    
    # Metric
    loss_fn = ELBOLoss(config)
    eval_fn = ELBOLoss(config)

    # Trainer & Evaluator
    trainer = Trainer(config = config,
                               data_loader = train_data_loader,
                               model = model,
                               optimizer = optimizer,
                               scheduler = scheduler,
                               loss_fn = loss_fn,
                               eval_fn = eval_fn)
    
    evaluator = Evaluator(config = config,
                                     data_loader = eval_data_loader,
                                     model = model,
                                     eval_fn = eval_fn)

    # Epoch loop
    start_epoch = 1
    best_error = 10000.
    for epoch in range(start_epoch, config.epochs+1):
        print(f'===== Start {epoch} epoch =====')
        
        # Training one epoch
        print("Training...")
        train_loss, train_val = trainer.train(epoch)

        # Logging
        logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
        logger.add_scalar('Loss(Reconstruction)/train', train_loss['Recon'], epoch)
        logger.add_scalar('Loss(KL_divergence)/train', train_loss['KL_div'], epoch)

        # evaluating
        if epoch % config.eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            if test_val['total'] < best_error:
                best_error = test_val['total']

                save_checkpoint('Saving the best model!',
                                os.path.join(model_dir, 'best.pth'),
                                epoch, 
                                best_error, 
                                model, 
                                optimizer, 
                                scheduler
                                )
            
            # Logging
            logger.add_scalar('Eval(total)/eval', test_val['total'], epoch)
            logger.add_scalar('Eval(recon.)/eval', test_val['Recon'], epoch)
            logger.add_scalar('Eval(KL_div.)/eval', test_val['KL_div'], epoch)

        # save the model
        if epoch % config.save_freq == 0:
            save_checkpoint('Saving...', 
                            os.path.join(model_dir, f'ckpt_epoch_{epoch}.pth'), 
                            epoch, 
                            best_error, 
                            model, 
                            optimizer, 
                            scheduler
                            )

        print(f'===== End {epoch} epoch =====')


if __name__=="__main__":
    config = Settings()
    main(config)