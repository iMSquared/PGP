import argparse
import os
import yaml
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

from load_q_policy import get_loader
from model import CVAE, ValueNet
from loss import ELBOLoss, ELBOLossImportanceSampling
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


class Settings():
    def __init__(self, params):    
        # ============================================================================================================================= 
        # Dataset
        self.path: str = '/home/share_folder/dataset/4.6'
        self.data_type = 'q_policy'
        if params.data_size == '1000':
            self.train_file = ('1000/sim/success',)
            self.test_file: tuple = ('1000/sim/success_eval',)
        elif params.data_size == '100':
            self.train_file = ('100/sim/success',)
            self.test_file: tuple = ('100/sim/success_eval',)
        elif params.data_size == '10':
            self.train_file = ('10/sim/success',)
            self.test_file: tuple = ('10/sim/success_eval',)
        
        self.randomize: bool = True
        
        self.batch_size: int = 64
        self.num_candidate: int = 64
        
        self.shuffle: bool = False # for using Sampler, it should be False
        self.use_sampler: bool = True
        self.max_len: int = 100
        self.seq_len: int = 31
        # |TODO| modify to automatically change
        self.dim_observation: int = 2
        self.dim_action: int = 2
        self.dim_state: int = 2
        self.dim_reward: int = 1

        # Architecture
        self.model = 'CVAE'
        self.optimizer: str = 'AdamW' # AdamW or AdamWR
        self.vae_beta = params.vae_beta

        self.dim_embed: int = 128
        self.dim_hidden: int = 128

        # for GPT
        self.dim_head: int = 128
        self.num_heads: int = 1
        self.dim_ffn: int = 128 * 4
        self.num_layers: int = 3

        # for CVAE
        self.latent_size: int = 16
        self.dim_condition: int = 128

        self.encoder_layer_sizes = [self.dim_embed, self.dim_embed + self.dim_condition, self.latent_size]
        self.decoder_layer_sizes = [self.latent_size, self.latent_size + self.dim_condition, self.dim_action]
        
        self.grad_clip: bool = True

        self.train_pos_en: bool = False
        self.use_reward: bool = False
        self.use_mask_padding: bool = True
        self.coefficient_loss: float = 1e-3

        self.dropout: float = 0.1
        self.action_tanh: bool = False

        # Training
        if params.cuda == '0':
            self.device: str = 'cuda:0' if th.cuda.is_available() else 'cpu'
        elif params.cuda == '1':
            self.device: str = 'cuda:1' if th.cuda.is_available() else 'cpu'
        else:
            self.device = 'cpu'
            
        self.resume: str = None # checkpoint file name for resuming
        # self.resume: str = 'ckpt_epoch_5500.pth'
        self.pre_trained: str = None
        # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
        self.epochs: int = 20000    # training step, not epochs
        # self.epochs: int = 1000    # epochs

        # Learning rate
        # |NOTE| using small learning rate, in order to apply warm up
        self.learning_rate: float = params.lr
        self.weight_decay: float = 1e-4
        self.warmup_step: int = int(5)       # epoch
        # For cosine annealing
        self.T_0: int = int(1e4)
        self.T_mult: int = 1
        self.lr_max: float = 0.01
        self.lr_mult: float = 0.5
        # self.lr_step = [500, 5000, 10000]
        self.lr_step = None

        # Logging
        self.exp_dir: str = '/home/share_folder/exp'
        # self.exp_dir: str = 'toy_domain/Learning/exp'
        self.model_name: str = params.model_name
        self.value_model_name: str = params.value_model_name
        self.value_model_ckpt: str = 'best.pth'

        self.print_freq: int = 100 # per train_steps
        self.train_eval_freq: int = 10 # per train_steps
        self.test_eval_freq: int = 10 # per train_steps
        self.save_freq: int = 100 # per train_steps
        # self.test_eval_freq: int = 1 # per epochs
        # self.save_freq: int = 10 # per epochs
        self.init_cnt_not_improve: int = 100

        self.log_para: bool = False
        self.log_grad: bool = False
        self.eff_grad: bool = False
        self.print_num_para: bool = True
        self.print_in_out: bool = False
        self.split_ratio: float = 0.8
        
        self.alpha_go: bool = False
        self.rank: bool = False
        self.preference_loss: bool = False
        self.preference_softmax: bool = False
        self.value_distribution: bool = False
        self.test_indicator: bool = False


def main(params):
    config = Settings(params)
    device = th.device(config.device)
    
    # |TODO| go to Setting()
    train_filename = config.train_file
    test_filename = config.test_file

    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)
    value_model_dir = os.path.join(config.exp_dir, config.value_model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    config_planner = {
        'model_name': config.model_name,
        'model_type': params.model_type,
        'num_plan': 100,
        # 'num_sim': [100, 80, 60, 40, 20, 10, 1],
        'num_sim': [100,],
        'ucb_const': 10
    }
    with open(os.path.join(model_dir, 'config_planner.yaml'), 'w') as f:
        yaml.dump(config_planner, f)

    logger = SummaryWriter(model_dir)
    
    train_dataset = []
    test_dataset = []
    for filename in train_filename:
        train_dataset += glob.glob(f'{dataset_path}/{filename}/*.pickle')
    for filename in test_filename:
        test_dataset += glob.glob(f'{dataset_path}/{filename}/*.pickle')
        
    # load checkpoint for value network
    value_model = ValueNet(config).to(device)
    value_model_optimizer = th.optim.AdamW(value_model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    filename = os.path.join(value_model_dir, config.value_model_ckpt)
    if os.path.isfile(filename):
        start_epoch, best_error, value_model, optimizer, scheduler = load_checkpoint(config, filename, value_model, value_model_optimizer, None)
        start_epoch += 1
        print("Loaded checkpoint '{}' (epoch {})".format(config.value_model_ckpt, start_epoch))
    else:
        raise Exception("No checkpoint found at '{}'".format(config.value_model_name))

    # generate dataloader
    train_loader = get_loader(config, train_dataset, value_model)
    test_loader = get_loader(config, test_dataset, value_model)

    print("# train data trajectories:", len(train_dataset))
    print("# train data steps:", len(train_loader.dataset))

    # model
    model = CVAE(config).to(device)

    # optimizer
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    # # learning rate scheduler
    # if config.optimizer == 'AdamW':
    #     scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 0.01))
    # elif config.optimizer == 'AdamWR':
    #     scheduler = CosineAnnealingWarmUpRestarts(
    #         optimizer=optimizer,
    #         T_0=config.T_0,
    #         T_mult=config.T_mult,
    #         eta_max=config.lr_max,
    #         T_up=config.warmup_step,
    #         gamma=config.lr_mult
    #     )
    # else:
    #     raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')
    if config.lr_step is not None:
        scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config.lr_step, gamma=config.lr_mult)
    else:
        scheduler = None

    # Metric
    loss_fn = ELBOLossImportanceSampling(config)
    eval_fn = ELBOLossImportanceSampling(config)

    # Trainer & Evaluator
    trainer = Trainer(config=config,
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    eval_fn=eval_fn)
    evaluator = Evaluator(config=config,
                        loader=test_loader,
                        model=model,
                        eval_fn=eval_fn)

    # save configuration
    # config.save(model_dir + '/config.yaml')
    # Logging model graph
    # dummy = next(iter(test_loader))
    # for k in dummy:
    #     dummy[k].to(device).detach()
    # logger.add_graph(ModelAsTuple(config, model), dummy)

    start_epoch = 1
    best_error = 10000.

    # load checkpoint for resuming
    if config.resume is not None:
        filename = os.path.join(model_dir, config.resume)
        if os.path.isfile(filename):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
            start_epoch += 1
            print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))

    # load checkpoint for pre-trained
    if config.pre_trained is not None:
        pre_trained_path = os.path.join(config.exp_dir, config.pre_trained)
        if os.path.isfile(pre_trained_path):
            start_epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, pre_trained_path, model, optimizer, scheduler)
            start_epoch = 1
            print("Loaded checkpoint '{}'".format(config.pre_trained))
        else:
            raise Exception("No checkpoint found at '{}'".format(config.resume))

    cnt_not_improve = config.init_cnt_not_improve
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
        if epoch % config.test_eval_freq == 0:
            print("Validating...")
            test_val = evaluator.eval(epoch)

            # save the best model
            # |TODO| change 'action' to 'total' @ trainer.py & evaluator.py -> merge 'CVAE' & others
            if config.model == 'CVAE' or config.model == 'ValueNet' or config.model == 'PolicyValueNet' or config.model == 'ValueNetDiscreteRepresentation':
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
                    cnt_not_improve = config.init_cnt_not_improve
                else:
                    cnt_not_improve -= 1
            else:
                if test_val['action'] < best_error:
                    best_error = test_val['action']

                    save_checkpoint('Saving the best model!',
                                    os.path.join(model_dir, 'best.pth'),
                                    epoch, 
                                    best_error, 
                                    model, 
                                    optimizer, 
                                    scheduler
                                    )
                    cnt_not_improve = config.init_cnt_not_improve
                else:
                    cnt_not_improve -= 1
            
            # Logging
            logger.add_scalar('Eval(total)/test', test_val['total'], epoch)
            logger.add_scalar('Eval(Reconstruction)/test', test_val['Recon'], epoch)
            logger.add_scalar('Eval(KL_divergence)/test', test_val['KL_div'], epoch)

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
        if cnt_not_improve < 0:
            print("Validation is not improved.")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--value_model_name", type=str, default="test")
    # parser.add_argument("--model_name", type=str, default="4.7_rank_1000_1")
    # parser.add_argument("--model_type", type=str, default="alphago")
    # parser.add_argument("--model_type", type=str, default="rank")
    # parser.add_argument("--model_type", type=str, default="preference")
    # parser.add_argument("--model_type", type=str, default="cvae")
    parser.add_argument("--model_type", type=str, default="q_policy")

    parser.add_argument("--data_size", type=str, default="1000")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vae_beta", type=float, default=1.0)
    parser.add_argument("--cuda", type=str, default="1")
    
    params = parser.parse_args()
    # wandb.init(project='LightDark-Value')
    # wandb.tensorboard.patch(save=False, pytorch=True)
    total_time_start = time.time()
    main(params)
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print("Total Time:", total_time)