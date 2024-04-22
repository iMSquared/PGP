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

from load import get_loader
from model import GPT2, RNN, LSTM, CVAE, ValueNet, ValueNetDiscreteRepresentation
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss, CQLLoss, RegressionLossValueWithNegativeData, RegressionLossValueWithNegativeDataEval, CustomLossValueDistribution, PreferenceLoss, RankLoss
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients


class Settings():
    def __init__(self, params):    
        # ============================================================================================================================= 
        # Dataset
        # self.path: str = 'toy_domain/Learning/dataset'
        self.path: str = '/home/share_folder/dataset/4.3/exec'
        if params.model_type == 'preference':
            self.path: str = '/home/share_folder/dataset/4.3/sim'
            self.data_type: str = 'preference'
            if params.data_size == 'big':
                self.train_file: tuple = ('success',) # folder name
                self.test_file: tuple = ('success_eval',) # folder name
                self.train_neg_file: tuple = ('fail')
                self.test_neg_file: tuple = ('fail_eval')
                self.pref_data_file: str = ''
            elif params.data_size == 'mini':
                self.train_file: tuple = ('success_mini',)
                self.test_file: tuple = ('success_mini_eval',)
                self.train_neg_file: tuple = ('fail_mini')
                self.test_neg_file: tuple = ('fail_mini_eval')
                self.pref_data_file: str = '_mini'
            elif params.data_size == 'tiny':
                self.train_file: tuple = ('success_tiny',)
                self.test_file: tuple = ('success_tiny_eval',)
                self.train_neg_file: tuple = ('fail_tiny')
                self.test_neg_file: tuple = ('fail_tiny_eval')
                self.pref_data_file: str = '_tiny'
        elif params.model_type == 'rank':
            self.path: str = '/home/share_folder/dataset/4.3/sim'
            self.data_type: str = 'success'
            if params.data_size == 'big':
                self.train_file: tuple = ('success',) # folder name
                self.test_file: tuple = ('success_eval',) # folder name
                self.train_neg_file: tuple = ('fail')
                self.test_neg_file: tuple = ('fail_eval')
            elif params.data_size == 'mini':
                self.train_file: tuple = ('success_mini',)
                self.test_file: tuple = ('success_mini_eval',)
                self.train_neg_file: tuple = ('fail_mini')
                self.test_neg_file: tuple = ('fail_mini_eval')
            elif params.data_size == 'tiny':
                self.train_file: tuple = ('success_tiny',)
                self.test_file: tuple = ('success_tiny_eval',)
                self.train_neg_file: tuple = ('fail_tiny')
                self.test_neg_file: tuple = ('fail_tiny_eval')
        elif params.model_type == 'alphago':
            self.path: str = '/home/share_folder/dataset/4.3/exec'
            self.data_type: str = 'success'
            if params.data_size == 'big':
                self.train_file: tuple = ('1000',) # folder name
                self.test_file: tuple = ('1000_eval',) # folder name
                self.train_neg_file: tuple = ('1000')
                self.test_neg_file: tuple = ('1000_eval')
            elif params.data_size == 'mini':
                self.train_file: tuple = ('100',)
                self.test_file: tuple = ('100_eval',)
                self.train_neg_file: tuple = ('100')
                self.test_neg_file: tuple = ('100_eval')
            elif params.data_size == 'tiny':
                self.train_file: tuple = ('10',)
                self.test_file: tuple = ('10_eval',)
                self.train_neg_file: tuple = ('10')
                self.test_neg_file: tuple = ('10_eval')
        # self.data_type: str = 'preference' # 'mcts' or 'success'
        # self.data_type_1: str = 'success' # 'mcts' or 'success'
        # self.data_type_2: str = 'mcts' # 'mcts' or 'success'
        # ============================================================================================================================= 
        self.randomize: bool = True
        self.filter: float = 51
        # ============================================================================================================================= 
        self.batch_size: int = 4096
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
        self.model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
        # self.model: str = 'ValueNetDiscreteRepresentation'
        
        self.optimizer: str = 'AdamW' # AdamW or AdamWR

        self.dim_embed: int = 128
        self.dim_hidden: int = 128
        # self.dim_embed: int = 32
        # self.dim_hidden: int = 32
        # self.dim_embed: int = 64
        # self.dim_hidden: int = 64
        # self.dim_embed: int = 128
        # self.dim_hidden: int = 128

        # for GPT
        self.dim_head: int = 128
        self.num_heads: int = 1
        self.dim_ffn: int = 128 * 4
        self.num_layers: int = 3
        # self.dim_head: int = 32
        # self.num_heads: int = 1
        # self.dim_ffn: int = 32 * 4
        # self.num_layers: int = 3
        # self.dim_head: int = 64
        # self.num_heads: int = 1
        # self.dim_ffn: int = 64 * 4
        # self.num_layers: int = 3
        # self.dim_head: int = 128
        # self.num_heads: int = 1
        # self.dim_ffn: int = 128 * 3
        # self.num_layers: int = 3

        # for CVAE
        self.latent_size: int = 16
        self.dim_condition: int = 128
        # self.latent_size: int = 32
        # self.dim_condition: int = 32
        # self.latent_size: int = 64
        # self.dim_condition: int = 64
        # self.latent_size: int = 128
        # self.dim_condition: int = 128
        self.encoder_layer_sizes = [self.dim_embed, self.dim_embed + self.dim_condition, self.latent_size]
        self.decoder_layer_sizes = [self.latent_size, self.latent_size + self.dim_condition, self.dim_action]
        # self.encoder_layer_sizes = [dim_embed, latent_size]
        # self.decoder_layer_sizes = [latent_size, dim_action]

        # =============================================================================================================================    
        # for discrete represented state
        self.node_size: int = 128
        self.category_size: int = 16
        self.class_size: int = 16
        
        # for CQL loss
        # self.cql_loss = 'mse'
        # self.cql_loss = 'mse2'
        # self.cql_loss = 'cql'
        # self.cql_reg: bool = 0.0
        # self.cql_logit_activation = False
        # self.cql_logit_activation = 'tanh'
        # self.cql_alpha = 1.0
        self.grad_clip: bool = True
        
        # for value normalization -1 ~ 1
        self.value_normalization: bool = False
        self.max_value = 100.0
        self.min_value = -30.0
        
        # for value distribution
        self.value_distribution: bool = False
        self.num_bin: int = 2
        self.bin_boundary: float = self.num_bin / 2
        self.custom_loss: bool = False
        self.loss_1: bool = False    # CE(V_\theta^+, v^+)
        self.loss_2: bool = False    # CE(V_\theta^-, v^-)
        self.loss_3: bool = False    # CE(V_\theta^OOD, v_min)
        self.loss_4: bool = False    # CE(V_\theta^+, v^_max|v^_min) (cf. after 4/1)
        self.loss_5: bool = False    # CE(V_\theta^-, v^_max)         (cf. after 4/1)
        self.loss_6: bool = False    # CE(V_\theta^-, v^_min)         (cf. after 4/1)
        
        if params.model_type == 'preference':
            self.alpha_go: bool = False
            self.rank: bool = False
            self.preference_loss: bool = True
            self.preference_softmax: bool = True
        elif params.model_type == 'rank':
            self.alpha_go: bool = False
            self.rank: bool = True
            self.preference_loss: bool = False
            self.preference_softmax: bool = False
        elif params.model_type == 'alphago':
            self.alpha_go: bool = True
            self.rank: bool = False
            self.preference_loss: bool = False
            self.preference_softmax: bool = False
        
        self.test_indicator: bool = False
        # =============================================================================================================================
        
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
        # self.resume: str = 'ckpt_epoch_600.pth'
        self.pre_trained: str = None
        # self.pre_trained: str = '4.17_CVAE/best.pth'
        # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
        self.epochs: int = 50000    # training step, not epochs
        # self.epochs: int = 1000    # epochs

        # Learning rate
        # |NOTE| using small learning rate, in order to apply warm up
        self.learning_rate: float = 1e-4
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
        # self.exp_dir: str = '/home/share_folder/exp'
        self.exp_dir: str = 'toy_domain/Learning/exp'
        self.model_name: str = params.model_name
        # self.model_name: str = 'test'
        # self.model_name: str = '4.17_ValueNet'
        # self.model_name: str = '3.26_CQL_without_reg'
        # self.model_name: str = '3.26_CQL_with_reg'
        # self.model_name: str = '3.27_cql_reg'
        # self.model_name: str = '3.27_cql_tanh'
        # self.model_name: str = '3.27_cql_reg_tanh'
        # self.model_name: str = '3.27_mse'
        # self.model_name: str = '3.27_mse2'
        # self.model_name: str = '3.27_mse2_reg'
        # self.model_name: str = '3.28_mse2_reg'
        # self.model_name: str = '3.28_mse2_tanh'
        # self.model_name: str = '3.28_mse2_grad_clip'
        # self.model_name: str = '3.28_mse2_tanh_grad_clip'
        # self.model_name: str = '3.28_mse2_lr_1e-4'
        # self.model_name: str = '3.28_mse2_lr_1e-3'
        # self.model_name: str = '3.28_mse2_dim128_plus1layer'
        # self.model_name: str = '3.28_mse2_dim128_plus1layer_tanh'
        # self.model_name: str = '3.28_mse2_dim128_class16'
        # self.model_name:str = '3.28_mse2_conti'
        # self.model_name:str = '3.28_mse2_conti_batch4096'
        # ============================================================
        # self.model_name: str = '3.28_mse2_dim128_batch4096'
        # self.model_name: str = '3.28_dreamer_batch4096'
        # self.model_name: str = '3.28_mse2_dim128_batch4096_bigdata'
        # self.model_name: str = '3.28_dreamer_dim128_batch4096_tanh_bigdata'
        # self.model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm'
        # self.model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm_bigdata'
        # self.model_name: str = '3.28_mse2_dim128_batch4096_tanh_value_norm'
        # self.model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm_reg'    
        # self.model_name: str = '3.28_dreamer_dim128_batch4096_tanh_reg'
        # self.model_name: str = '3.29_mse2_dim128_batch4096_dist'
        # self.model_name: str = '3.29_mse2_dim128_batch4096_dist_bigdata'
        # self.model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse'
        # self.model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse_bigdata_re'
        # self.model_name: str = '3.30_mse2_dim128_batch4096_dist_eval_mse_bin50'
        # self.model_name: str = '3.30_mse2_dim128_batch4096_dist_bin50_loss45'
        # self.model_name: str = '3.30_mse2_dim128_batch4096_dist_bin50_loss1245'
        # self.model_name: str = '3.30_dreamer_dim128_batch4096_indicator_mse2'
        # self.model_name: str = '3.30_dreamer_dim128_batch4096_indicator_cql'
        # self.model_name = '4.1_vd_len30'
        # self.model_name = '4.1_rank_len30'
        # self.model_name = '4.1_vd_len4'
        # self.model_name = '4.1_rank_len4'
        # self.model_name = '4.2_vd_len30_bigdata'
        # self.model_name = '4.2_rank_len30_bigdata'
        # self.model_name = '4.2_vd_len4_bf'
        # self.model_name = '4.2_rank_len4_bf'
        # self.model_name = '4.3_vd_len4_bf'
        # self.model_name = '4.3_rank_len4_bf'
        # self.model_name = '4.3_rank_len4_bf_scale2'
        # self.model_name = '4.3_rank_len4_bf_scale0.5'
        # self.model_name = '4.3_rank_len4_bf_fix'
        # self.model_name = '4.3_binary_classification_SorF'
        # self.model_name = f'4.3_preference_sigmoid_steplr_sim'
        # self.model_name = '4.3_alphago_sim_mini'
        # self.model_name = '4.4_preference_sigmoid_1'
        # self.model_name = '4.4_preference_sigmoid_2'
        # self.model_name = '4.4_preference_softmax_1'
        # self.model_name = '4.4_preference_softmax_2'
        # self.model_name = '4.4_preference_sigmoid_1_tiny'
        # self.model_name = '4.4_preference_sigmoid_2_tiny'
        # self.model_name = '4.4_preference_softmax_1_tiny'
        # self.model_name = '4.4_preference_softmax_2_tiny'
        # self.model_name = '4.4_rank_2'
        # self.model_name = '4.4_rank_3'
        # self.model_name = '4.4_rank_tiny_1'
        # self.model_name = '4.4_rank_tiny_2'
        # self.model_name = '4.4_rank_mini_1'
        # self.model_name = '4.4_rank_mini_2'
        # self.model_name = '4.4_alphago_1000_1'
        # self.model_name = '4.4_alphago_1000_2'
        # self.model_name = '4.4_alphago_1000_3'
        # self.model_name = '4.4_preference_softmax_3'
        # self.model_name = '4.4_preference_sigmoid_3_tiny'
        # self.model_name = '4.4_preference_softmax_3_tiny'
        # self.model_name = '4.4_alphago_100_1'
        # self.model_name = '4.4_alphago_100_2'
        # self.model_name = '4.4_alphago_100_3'
        # self.model_name = '4.4_alphago_10_1'
        # self.model_name = '4.4_alphago_10_2'
        # self.model_name = '4.4_alphago_10_3'

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


def main(params):
    config = Settings(params)
    # |TODO| go to Setting()
    train_filename = config.train_file
    test_filename = config.test_file
    # if config.cql_loss == 'mse2' or (config.custom_loss and (config.loss_6 or config.loss_2)) or config.preference_loss:
    if (config.custom_loss and (config.loss_6 or config.loss_2)) or config.preference_loss or config.rank:
        train_neg_filename = config.train_neg_file
        test_neg_filename = config.test_neg_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    config_planner = {
        'model_name': config.model_name,
        'model_type': params.model_type,
        'num_plan': 100,
        # 'num_sim': [100, 80, 60, 40, 20, 10, 1],
        'num_sim': [500,],
    }
    if params.model_type == 'preference':
        config_planner['ucb_const'] = 150
    elif params.model_type == 'rank' or params.model_type == 'alphago':
        config_planner['ucb_const'] = 30
    with open(os.path.join(model_dir, 'config_planner.yaml'), 'w') as f:
        yaml.dump(config_planner, f)

    logger = SummaryWriter(model_dir)
    
    train_dataset = []
    test_dataset = []
    for filename in train_filename:
        train_dataset += glob.glob(f'{dataset_path}/{filename}/*.pickle')
    for filename in test_filename:
        test_dataset += glob.glob(f'{dataset_path}/{filename}/*.pickle')
    # if config.cql_loss == 'mse2' or (config.custom_loss and (config.loss_6 or config.loss_2)):
    if (config.custom_loss and (config.loss_6 or config.loss_2)) or config.rank:
        neg_train_dataset = []
        neg_test_dataset = []
        neg_train_dataset += glob.glob(f'{dataset_path}/{train_neg_filename}/*.pickle')
        neg_test_dataset += glob.glob(f'{dataset_path}/{test_neg_filename}/*.pickle')
    

    # generate dataloader
    if config.preference_loss:
        train_loader = get_loader(config, train_dataset, train=True)
        test_loader = get_loader(config, test_dataset, train=False)
    else:
        train_loader = get_loader(config, train_dataset)
        test_loader = get_loader(config, test_dataset)
    # if config.cql_loss == 'mse2' or (config.custom_loss and (config.loss_6 or config.loss_2)) or config.preference_loss:
    if (config.custom_loss and (config.loss_6 or config.loss_2)) or config.rank:
        neg_train_loader = get_loader(config, neg_train_dataset)
        neg_test_loader = get_loader(config, neg_test_dataset)
        
    print("# train data trajectories:", len(train_dataset))
    print("# train data steps:", len(train_loader.dataset))

    # model
    device = th.device(config.device)
    if config.model == 'GPT':
        model = GPT2(config).to(device)
    elif config.model == 'RNN':
        model = RNN(config).to(device)
    elif config.model == 'LSTM':
        model = LSTM(config).to(device)
    elif config.model == 'CVAE' or config.model == 'PolicyValueNet':
        model = CVAE(config).to(device)
    elif config.model == 'ValueNet':
        model = ValueNet(config).to(device)
    elif config.model == 'ValueNetDiscreteRepresentation':
        model = ValueNetDiscreteRepresentation(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", "LSTM", "CVAE", "ValueNet", or "PolicyValueNet.')

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
    # |TODO| implement Chamfer distance
    if config.model == 'CVAE':
        loss_fn = ELBOLoss(config)
        eval_fn = ELBOLoss(config)
    elif config.model == 'ValueNet':
        if config.preference_loss:
            loss_fn = PreferenceLoss(config)
            eval_fn = PreferenceLoss(config)
        elif config.rank:
            loss_fn = RankLoss(config)
            eval_fn = RankLoss(config)
        elif config.alpha_go:
            loss_fn = RegressionLossValue(config)
            eval_fn = RegressionLossValue(config)
        elif config.value_distribution and config.custom_loss:
            loss_fn = CustomLossValueDistribution(config)
            eval_fn = CustomLossValueDistribution(config)
        # elif config.cql_loss == 'mse2':
        #     loss_fn = RegressionLossValueWithNegativeData(config)
        #     # if config.value_distribution:
        #     #     eval_fn = RegressionLossValueWithNegativeDataEval(config)
        #     # else:
        #     eval_fn = RegressionLossValueWithNegativeData(config)
        else:
            loss_fn = RegressionLossValue(config)
            eval_fn = RegressionLossValue(config)
    elif config.model == 'PolicyValueNet':
        loss_fn = None
        eval_fn = None
    elif config.model == 'ValueNetDiscreteRepresentation':
        if config.cql_loss == 'cql':
            loss_fn = CQLLoss(config)
            eval_fn = CQLLoss(config)
        elif config.cql_loss == 'mse2':
            loss_fn = RegressionLossValueWithNegativeData(config)
            eval_fn = RegressionLossValueWithNegativeData(config)
        else:
            loss_fn = RegressionLossValue(config)
            eval_fn = RegressionLossValue(config)
    else:
        loss_fn = RegressionLossPolicy(config)
        eval_fn = RegressionLossPolicy(config)

    # Trainer & Evaluator
    # if config.cql_loss == 'mse2' or (config.custom_loss and (config.loss_6 or config.loss_2)) or config.preference_loss:
    if (config.custom_loss and (config.loss_6 or config.loss_2)) or config.rank:
        trainer = Trainer(config=config,
                        loader=train_loader,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=loss_fn,
                        eval_fn=eval_fn,
                        neg_loader=neg_train_loader)
        evaluator = Evaluator(config=config,
                            loader=test_loader,
                            model=model,
                            eval_fn=eval_fn,
                            neg_loader=neg_test_loader)
    else:
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
        if config.model == 'CVAE':
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(Reconstruction)/train', train_loss['Recon'], epoch)
            logger.add_scalar('Loss(KL_divergence)/train', train_loss['KL_div'], epoch)
        elif config.model == 'ValueNet':
            if config.preference_loss or config.alpha_go or config.rank:
                logger.add_scalar('Train(total)', train_loss['total'], epoch)
                if config.lr_step is not None:
                    logger.add_scalar('LR', trainer.scheduler.get_last_lr()[0], epoch)
            elif config.value_distribution and config.custom_loss:
                logger.add_scalar('Train(total)', train_loss['total'], epoch)
                if config.loss_1:
                    logger.add_scalar('Train(loss1)', train_loss['loss_1'], epoch)
                if config.loss_2:
                    logger.add_scalar('Train(loss2)', train_loss['loss_2'], epoch)
                if config.loss_3:
                    logger.add_scalar('Train(loss3)', train_loss['loss_3'], epoch)
                if config.loss_4:
                    logger.add_scalar('Train(loss4)', train_loss['loss_4'], epoch)
                if config.loss_5:
                    logger.add_scalar('Train(loss5)', train_loss['loss_5'], epoch)
                if config.loss_6:
                    logger.add_scalar('Train(loss6)', train_loss['loss_6'], epoch)
            elif config.cql_loss == 'mse2':
                logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
                logger.add_scalar('Loss(Positive)/train', train_loss['pos'], epoch)
                logger.add_scalar('Loss(Negative)/train', train_loss['neg'], epoch)
            else:
                logger.add_scalar('Loss/train', train_loss['total'], epoch)
        elif config.model == 'PolicyValueNet':
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(action)/train', train_loss['action'], epoch)
            logger.add_scalar('Loss(accumulated reward)/train', train_loss['accumulated_reward'], epoch)
            # logger.add_scalar('Eval(action)/train', train_val['action'], epoch)
        elif config.model == 'ValueNetDiscreteRepresentation':
            if config.cql_loss == 'cql':
                logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
                logger.add_scalar('Loss(MSE)/train', train_loss['accumulated_reward'], epoch)
                logger.add_scalar('Loss(Conservative)/train', train_loss['conservative'], epoch)
            elif config.cql_loss == 'mse2':
                logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
                logger.add_scalar('Loss(Positive)/train', train_loss['pos'], epoch)
                logger.add_scalar('Loss(Negative)/train', train_loss['neg'], epoch)
            else:
                logger.add_scalar('Loss/train', train_loss['total'], epoch)
        else:
            logger.add_scalar('Loss(total)/train', train_loss['total'], epoch)
            logger.add_scalar('Loss(action)/train', train_loss['action'], epoch)
            # if config.use_reward:
            #     logger.add_scalar('Loss(reward)/train', train_loss['reward'], epoch)

            # logger.add_scalar('Eval(action)/train', train_val['action'], epoch)
            # if config.use_reward:
            #     logger.add_scalar('Eval(reward)/train', train_val['reward'], epoch)

        # |FIXME| debug for eff_grad: "RuntimeError: Boolean value of Tensor with more than one value is ambiguous"
        # log_gradients(model, logger, epoch, log_grad=config.log_grad, log_param=config.log_para, eff_grad=config.eff_grad, print_num_para=config.print_num_para)

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
            if config.model == 'CVAE':
                logger.add_scalar('Eval(total)/test', test_val['total'], epoch)
                logger.add_scalar('Eval(Reconstruction)/test', test_val['Recon'], epoch)
                logger.add_scalar('Eval(KL_divergence)/test', test_val['KL_div'], epoch)
            elif config.model == 'ValueNet':
                if config.preference_loss or config.alpha_go or config.rank:
                    logger.add_scalar('Eval/test', test_val['total'], epoch)
                elif config.value_distribution and config.custom_loss:
                    logger.add_scalar('Eval/test', test_val['total'], epoch)
                    if config.loss_1:
                        logger.add_scalar('Eval(Loss1)', test_val['loss_1'], epoch)
                    if config.loss_2:
                        logger.add_scalar('Eval(Loss2)', test_val['loss_2'], epoch)
                    if config.loss_3:
                        logger.add_scalar('Eval(Loss3)', test_val['loss_3'], epoch)
                    if config.loss_4:
                        logger.add_scalar('Eval(Loss4)', test_val['loss_4'], epoch)
                    if config.loss_5:
                        logger.add_scalar('Eval(Loss5)', test_val['loss_5'], epoch)
                    if config.loss_6:
                        logger.add_scalar('Eval(Loss6)', test_val['loss_6'], epoch)
                elif config.cql_loss == 'mse2':
                    logger.add_scalar('Eval(total)/train', test_val['total'], epoch)
                    logger.add_scalar('Eval(Positive)/train', test_val['pos'], epoch)
                    logger.add_scalar('Eval(Negative)/train', test_val['neg'], epoch)
                    if config.value_distribution:
                        logger.add_scalar('Eval(PositiveMSE)/train', test_val['pos_mse'], epoch)
                        logger.add_scalar('Eval(NegativeMSE)/train', test_val['neg_mse'], epoch)
                else:
                    logger.add_scalar('Eval/test', test_val['total'], epoch)
            elif config.model == 'PolicyValueNet':
                logger.add_scalar('Eval(total)/test', test_val['total'], epoch)
                logger.add_scalar('Eval(action)/test', test_val['action'], epoch)
                logger.add_scalar('Eval(accumulated reward)/test', test_val['accumulated_reward'], epoch)
            elif config.model == 'ValueNetDiscreteRepresentation':
                if config.cql_loss == 'cql':
                    logger.add_scalar('Eval(total)/train', test_val['total'], epoch)
                    logger.add_scalar('Eval(MSE)/train', test_val['accumulated_reward'], epoch)
                    logger.add_scalar('Eval(Conservative)/train', test_val['conservative'], epoch)
                elif config.cql_loss == 'mse2':
                    logger.add_scalar('Eval(total)/train', test_val['total'], epoch)
                    logger.add_scalar('Eval(Positive)/train', test_val['pos'], epoch)
                    logger.add_scalar('Eval(Negative)/train', test_val['neg'], epoch)
                else:
                    logger.add_scalar('Eval/test', test_val['total'], epoch)
            else:
                logger.add_scalar('Eval(action)/test', test_val['action'], epoch)
                # if config.use_reward:
                #     logger.add_scalar('Eval(reward)/test', test_val['reward'], epoch)
        
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
            print("Validation is not improvement.")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--model_type", type=str, default="alphago")
    # parser.add_argument("--model_type", type=str, default="rank")
    # parser.add_argument("--model_type", type=str, default="preference")
    parser.add_argument("--data_size", type=str, default="tiny")
    parser.add_argument("--cuda", type=str, default="0")
    
    params = parser.parse_args()
    # wandb.init(project='LightDark-Value')
    # wandb.tensorboard.patch(save=False, pytorch=True)
    total_time_start = time.time()
    main(params)
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print("Total Time:", total_time)