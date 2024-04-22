import os
import glob
import time
import shutil
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from typing import List
import pickle
import torch as th
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.ticker as ticker
# import wandb
from torch.utils.data import DataLoader
from load import get_loader, BatchMaker
from model import GPT2, RNN, LSTM, CVAE, ValueNet, ValueNetDiscreteRepresentation
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss, CQLLoss, RegressionLossValueWithNegativeData
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients
from run import *
from load import get_loader, LightDarkDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F

class RANK_Settings1(Serializable):
    comparison_traj_num:int = 25
    train_len = 30
    # Dataset
    path: str = '/home/share_folder/dataset/eval'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 0
    # test_file = ['testset/success', 'testset/fail']
    test_file = ['len4_eval']
    # train_file_1: str = 'sim_success_exp_const_30_std0.5'
    # train_file_2: str = 'mcts_1_exp_const_30_std0.5'
    # test_file: str = 'sim_success_randomize_2'
    batch_size: int = 1
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
    # model: str = 'ValueNetDiscreteRepresentation'
    
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 128
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]
    
        # =============================================================================================================================    
    # for discrete represented state
    node_size: int = 128
    category_size: int = 16
    class_size: int = 16
    
    # for CQL loss
    # cql_loss = 'mse'
    cql_loss = 'mse2'
    # cql_loss = 'cql'
    cql_reg: bool = 0.0
    cql_logit_activation = False
    # cql_logit_activation = 'tanh'
    cql_alpha = 1.0
    grad_clip: bool = True
    
    # for value normalization -1 ~ 1
    value_normalization: bool = False
    max_value = 100.0
    min_value = -30.0
    
    # for value distribution
    value_distribution: bool = True
    num_bin = 50
    custom_loss = True
    loss_1: bool = True    # CE(V_\theta^+, v^+)
    loss_2: bool = True   # CE(V_\theta^-, v^-)
    loss_3: bool = False    # CE(V_\theta^OOD, v_min)
    loss_4: bool = False    # CE(V_\theta^+, v^_max)
    loss_5: bool = False    # CE(V_\theta^-, v^_min)
    
    test_indicator: bool = False
    # =============================================================================================================================

    train_pos_en: bool = False
    use_reward: bool = False
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    pre_trained: str = None
    epochs: int = 1000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_step: int = int(5)       # epoch
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = '/home/share_folder/exp'
    model_name: str = f'4.1_rank_len{train_len}'

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    
    num_bin:int = 50
    comparison_traj_num:int = 2
    compare_threshold:float = 0.0001
    save_traj:str = f'/home/pomdp2/Projects/POMDP/toy_domain/Learning/dist_len{train_len}_step{comparison_traj_num}'

class DV_Settings2(Serializable):
    train_len = 30
    # Dataset
    path: str = '/home/share_folder/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 0
    test_file = ['testset/success', 'testset/fail']
    # train_file_1: str = 'sim_success_exp_const_30_std0.5'
    # train_file_2: str = 'mcts_1_exp_const_30_std0.5'
    # test_file: str = 'sim_success_randomize_2'
    batch_size: int = 1
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
    # model: str = 'ValueNetDiscreteRepresentation'
    
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 128
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]
    
        # =============================================================================================================================    
    # for discrete represented state
    node_size: int = 128
    category_size: int = 16
    class_size: int = 16
    
    # for CQL loss
    # cql_loss = 'mse'
    cql_loss = 'mse2'
    # cql_loss = 'cql'
    cql_reg: bool = 0.0
    cql_logit_activation = False
    # cql_logit_activation = 'tanh'
    cql_alpha = 1.0
    grad_clip: bool = True
    
    # for value normalization -1 ~ 1
    value_normalization: bool = False
    max_value = 100.0
    min_value = -30.0
    
    # for value distribution
    value_distribution: bool = True
    num_bin = 50
    custom_loss = True
    loss_1: bool = False    # CE(V_\theta^+, v^+)
    loss_2: bool = False   # CE(V_\theta^-, v^-)
    loss_3: bool = False    # CE(V_\theta^OOD, v_min)
    loss_4: bool = True    # CE(V_\theta^+, v^_max)
    loss_5: bool = True    # CE(V_\theta^-, v^_min)
    
    test_indicator: bool = False
    # =============================================================================================================================

    train_pos_en: bool = False
    use_reward: bool = False
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    pre_trained: str = None
    epochs: int = 1000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_step: int = int(5)       # epoch
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = '/home/share_folder/exp'
    model_name: str = f'4.1_vd_len{train_len}'

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    
    num_bin:int = 50
    compare_threshold:float = 0.0001
    save_traj:str = '/home/pomdp/workspace/POMDP/toy_domain/Learning'
    
class RANK_Settings3(Serializable):
    comparison_traj_num:int = 29
    train_len = 30
    # Dataset
    path: str = '/home/share_folder/dataset/eval'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 0
    test_file = ['eval']
    # train_file_1: str = 'sim_success_exp_const_30_std0.5'
    # train_file_2: str = 'mcts_1_exp_const_30_std0.5'
    # test_file: str = 'sim_success_randomize_2'
    batch_size: int = 1
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
    # model: str = 'ValueNetDiscreteRepresentation'
    
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 128
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]
    
        # =============================================================================================================================    
    # for discrete represented state
    node_size: int = 128
    category_size: int = 16
    class_size: int = 16
    
    # for CQL loss
    # cql_loss = 'mse'
    cql_loss = 'mse2'
    # cql_loss = 'cql'
    cql_reg: bool = 0.0
    cql_logit_activation = False
    # cql_logit_activation = 'tanh'
    cql_alpha = 1.0
    grad_clip: bool = True
    
    # for value normalization -1 ~ 1
    value_normalization: bool = False
    max_value = 100.0
    min_value = -30.0
    
    # for value distribution
    value_distribution: bool = True
    num_bin = 50
    custom_loss = True
    loss_1: bool = True    # CE(V_\theta^+, v^+)
    loss_2: bool = True   # CE(V_\theta^-, v^-)
    loss_3: bool = False    # CE(V_\theta^OOD, v_min)
    loss_4: bool = False    # CE(V_\theta^+, v^_max)
    loss_5: bool = False    # CE(V_\theta^-, v^_min)
    
    test_indicator: bool = False
    # =============================================================================================================================

    train_pos_en: bool = False
    use_reward: bool = False
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    pre_trained: str = None
    epochs: int = 1000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_step: int = int(5)       # epoch
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = '/home/share_folder/exp'
    model_name: str = f'3.30_mse2_dim128_batch4096_dist_bin50_loss45'

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    
    num_bin:int = 50
    compare_threshold:float = 0.0001
    save_traj:str = f'/home/pomdp2/Projects/POMDP/toy_domain/Learning/dist_len{train_len}_step{comparison_traj_num}'

class DV_Settings4(Serializable):
    train_len = 30
    # Dataset
    path: str = '/home/share_folder/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 0
    test_file = ['testset/success', 'testset/fail']
    # train_file_1: str = 'sim_success_exp_const_30_std0.5'
    # train_file_2: str = 'mcts_1_exp_const_30_std0.5'
    # test_file: str = 'sim_success_randomize_2'
    batch_size: int = 1
    shuffle: bool = True # for using Sampler, it should be False
    use_sampler: bool = False
    max_len: int = 100
    seq_len: int = 31
    # |TODO| modify to automatically change
    dim_observation: int = 2
    dim_action: int = 2
    dim_state: int = 2
    dim_reward: int = 1

    # Architecture
    model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
    # model: str = 'ValueNetDiscreteRepresentation'
    
    optimizer: str = 'AdamW' # AdamW or AdamWR

    dim_embed: int = 128
    dim_hidden: int = 128
    # dim_embed: int = 32
    # dim_hidden: int = 32
    # dim_embed: int = 64
    # dim_hidden: int = 64
    # dim_embed: int = 128
    # dim_hidden: int = 128

    # for GPT
    dim_head: int = 128
    num_heads: int = 1
    dim_ffn: int = 128 * 4
    num_layers: int = 3
    # dim_head: int = 32
    # num_heads: int = 1
    # dim_ffn: int = 32 * 4
    # num_layers: int = 3
    # dim_head: int = 64
    # num_heads: int = 1
    # dim_ffn: int = 64 * 4
    # num_layers: int = 3
    # dim_head: int = 128
    # num_heads: int = 1
    # dim_ffn: int = 128 * 3
    # num_layers: int = 3

    # for CVAE
    latent_size: int = 16
    dim_condition: int = 128
    # latent_size: int = 32
    # dim_condition: int = 32
    # latent_size: int = 64
    # dim_condition: int = 64
    # latent_size: int = 128
    # dim_condition: int = 128
    encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
    decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
    # encoder_layer_sizes = [dim_embed, latent_size]
    # decoder_layer_sizes = [latent_size, dim_action]
    
        # =============================================================================================================================    
    # for discrete represented state
    node_size: int = 128
    category_size: int = 16
    class_size: int = 16
    
    # for CQL loss
    # cql_loss = 'mse'
    cql_loss = 'mse2'
    # cql_loss = 'cql'
    cql_reg: bool = 0.0
    cql_logit_activation = False
    # cql_logit_activation = 'tanh'
    cql_alpha = 1.0
    grad_clip: bool = True
    
    # for value normalization -1 ~ 1
    value_normalization: bool = False
    max_value = 100.0
    min_value = -30.0
    
    # for value distribution
    value_distribution: bool = True
    num_bin = 50
    custom_loss = True
    loss_1: bool = False    # CE(V_\theta^+, v^+)
    loss_2: bool = False   # CE(V_\theta^-, v^-)
    loss_3: bool = False    # CE(V_\theta^OOD, v_min)
    loss_4: bool = True    # CE(V_\theta^+, v^_max)
    loss_5: bool = True    # CE(V_\theta^-, v^_min)
    
    test_indicator: bool = False
    # =============================================================================================================================

    train_pos_en: bool = False
    use_reward: bool = False
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cpu'
    resume: str = 'best.pth' # checkpoint file name for resuming
    pre_trained: str = None
    epochs: int = 1000

    # Learning rate
    # |NOTE| using small learning rate, in order to apply warm up
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_step: int = int(5)       # epoch
    # For cosine annealing
    T_0: int = int(1e4)
    T_mult: int = 1
    lr_max: float = 0.01
    lr_mult: float = 0.5

    # Logging
    exp_dir: str = '/home/share_folder/exp'
    model_name: str = '3.30_mse2_dim128_batch4096_dist_eval_mse_bin50'

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    
    num_bin:int = 50
    compare_threshold:float = 0.0001
    save_traj:str = '/home/pomdp/workspace/POMDP/toy_domain/Learning'
    
def eval(config, model, dataset, data_loader, neg_dataloader=None):        
    # dst_dir = os.path.join(os.getcwd(), config.path, 'sim_3.25_test/out')
    
    model.eval()
    if str(config.device) == 'cuda':
        th.cuda.empty_cache()

    with th.no_grad():
        end = time.time()
        
        test_result = {
            'target': [],
            'output': []
        }
        for i, data in enumerate(data_loader):
            target = {}
            target_action = th.squeeze(data['next_action'])
            target_value = th.squeeze(data['accumulated_reward'])
            target['action'] = target_action
            target['accumulated_reward'] = target_value
            target['value'] = data['value']
            
            if config.filter:
                if target_value.item() > config.filter:
                    # shutil.move(dataset[i], dst_dir)
                    continue

            if config.model == 'CVAE':
                recon_x, mean, log_var, z = model(data)

            elif config.model == 'ValueNet':
                if config.value_distribution:
                    pred = model.inference(data)
                else:
                    pred = model(data)

            elif config.model == 'PolicyValueNet':
                value, recon_x, mean, log_var, z = model(data)

            elif config.model == 'ValueNetDiscreteRepresentation':
                if config.cql_loss == 'cql':
                    pred, pred_rand, entropy = model(data)
                else:
                    pred, entropy = model(data)
            else:
                pred = model(data)
            
            if config.value_distribution:
                test_result['target'].append(target['value'].detach().item())
                # test_result['output'].append(pred[1].detach().item())
                test_result['output'].append(pred[0].detach().numpy())
            else:
                test_result['target'] += [target['accumulated_reward'].detach()]
                test_result['output'] += [pred.squeeze().detach()]
                
        
    return test_result


def main():
    config1 = RANK_Settings1()
    config2 = DV_Settings2()
    config3 = RANK_Settings3()
    config4 = DV_Settings4()
    dataset_filename = config1.test_file
    dataset_path = config1.path
    
    if not os.path.exists(config1.exp_dir):
        os.mkdir(config1.exp_dir)
    model_dir1 = os.path.join(config1.exp_dir, config1.model_name)
    model_dir2 = os.path.join(config2.exp_dir, config2.model_name)
    model_dir3 = os.path.join(config3.exp_dir, config3.model_name)
    model_dir4 = os.path.join(config4.exp_dir, config4.model_name)

    # model
    device = th.device(config1.device)
    if config1.model == 'ValueNet':
        model1 = ValueNet(config1).to(device)
        model2 = ValueNet(config2).to(device)
        model3 = ValueNet(config3).to(device)
        model4 = ValueNet(config4).to(device)
    elif config1.model == 'ValueNetDiscreteRepresentation':
        model = ValueNetDiscreteRepresentation(config1).to(device)
    else:
        raise Exception(f'"{config1.model}" is not support!! You should select "GPT", "RNN", or "LSTM".')

    # optimizer
    optimizer1 = th.optim.AdamW(model1.parameters(),
                               lr=config1.learning_rate,
                               weight_decay=config1.weight_decay)
    
    optimizer2 = th.optim.AdamW(model2.parameters(),
                               lr=config1.learning_rate,
                               weight_decay=config1.weight_decay)

    scheduler = None    

    # load checkpoint for resuming
    filename1 = os.path.join(model_dir1, config1.resume)
    filename2 = os.path.join(model_dir2, config2.resume)
    filename3 = os.path.join(model_dir3, config3.resume)
    filename4 = os.path.join(model_dir4, config4.resume)
    epoch1, best_error, model1, optimizer, scheduler = load_checkpoint(config1, filename1, model1, optimizer1, scheduler)
    epoch2, best_error, model2, optimizer, scheduler = load_checkpoint(config2, filename2, model2, optimizer2, scheduler)
    epoch3, best_error, model3, optimizer, scheduler = load_checkpoint(config3, filename3, model3, optimizer1, scheduler)
    epoch4, best_error, mode42, optimizer, scheduler = load_checkpoint(config4, filename4, model4, optimizer2, scheduler)
    print("Loaded checkpoint '{}' (epoch {})".format(config1.resume, epoch1))
    print("Loaded checkpoint '{}' (epoch {})".format(config2.resume, epoch2))
    print("Loaded checkpoint '{}' (epoch {})".format(config3.resume, epoch3))
    print("Loaded checkpoint '{}' (epoch {})".format(config4.resume, epoch4))
    same_traj(config1, dataset_path, config1.comparison_traj_num, model1, model2, model3, model4)    
    return

def same_traj(config, data_folder_path, comparison_traj_num, model1, model2, model3, model4):
    compare_threshold = config.compare_threshold
    sample_traj_num=200
    dataset =  glob.glob(f'{data_folder_path}/*.pickle')
    test = dataset[sample_traj_num:]
    sample_traj = dataset[:sample_traj_num]
    test_dataset = LightDarkDataset(config, test, None)
    sample_dataset = LightDarkDataset(config, sample_traj, None)

    # Get comparison dataset
    test_np_obs = np.inf * np.ones((len(test_dataset), 2, comparison_traj_num))
    test_np_action = np.inf * np.ones((len(test_dataset), 2, comparison_traj_num))
    
    print("Now reading all dataset...")
    for i in range(len(test)):
        try:
            traj_len = np.array(test_dataset[i]['observation']).shape[0]
        except IndexError:
            breakpoint()
        if traj_len < comparison_traj_num:
            test_np_obs[i][:, :traj_len] = np.transpose(np.array(test_dataset[i]['observation']))
            test_np_action[i][:, :traj_len] = np.transpose(np.array(test_dataset[i]['action']))
        else:
            test_np_obs[i] = np.transpose(np.array(test_dataset[i]['observation']), (1,0))[:,:comparison_traj_num]
            test_np_action[i] = np.transpose(np.array(test_dataset[i]['action']), (1,0))[:,:comparison_traj_num]
        
    print("Check the same trajectories...")
    for i in range(len(sample_dataset)):
        
        if np.array(sample_dataset[i]['observation']).shape[0] > comparison_traj_num:
            # Compare observation and action
            sample_traj_obs = np.expand_dims(np.transpose(np.array(sample_dataset[i]['observation']), (1,0))[:,:comparison_traj_num], axis=0)
            sample_traj_action = np.expand_dims(np.transpose(np.array(sample_dataset[i]['action']), (1,0))[:,:comparison_traj_num], axis=0)
            tmp_comparison_obs = (np.abs(sample_traj_obs - test_np_obs) > compare_threshold).sum(axis=2).sum(axis=1)
            tmp_comparison_action = (np.abs(sample_traj_action - test_np_action) > compare_threshold).sum(axis=2).sum(axis=1)
            same_obs = np.argwhere(tmp_comparison_obs==0)
            same_action = np.argwhere(tmp_comparison_action==0)
            final = []
            final_history = []
            
            for j in same_obs:
                if j in same_action:
                    final.append(int((np.sum(test_dataset[j.item()]['reward'])-(-30))/130*config.num_bin))
                    final_history.append(test_dataset[j.item()])
            if len(final) > 3:    
                keys = [i for i in range(config.num_bin)]
                fig, axes = plt.subplots(3,2)
                values = [0 for i in range(config.num_bin)]
                count_values = np.unique(final, return_counts=True)
                for j, l in enumerate(count_values[0]):
                    values[l] = count_values[1][j]
                axes[0][1].set_title(f'GT({len(final)}) -multi')
                axes[0][1].bar(keys, np.array(values)/len(final))
                axes[0][0].set_title(f'GT({len(final)}) -binary')
                axes[0][0].bar([0,1], [sum(np.array(np.array(values)/len(final))[:int(config.num_bin/2)]), sum(np.array(np.array(values)/len(final))[int(config.num_bin/2):])])
                plot(axes[0][1], config.num_bin)
                plot(axes[0][0], 2)
                
                use_input={}
                use_input['action']= sample_dataset[i]['action'][:comparison_traj_num]
                use_input['observation'] = sample_dataset[i]['observation'][:comparison_traj_num]
                use_input['next_state'] = sample_dataset[i]['next_state'][:comparison_traj_num]
                use_input['reward'] = sample_dataset[i]['reward'][:comparison_traj_num]
                use_input['goal_state'] = sample_dataset[i]['goal_state']
                use_input['traj_len'] = comparison_traj_num
                batcher = BatchMaker(config)
                loader = DataLoader([use_input],
                                    batch_size=config.batch_size,
                                    shuffle=config.shuffle,
                                    sampler= None,
                                    collate_fn=batcher)
                test_result1 = eval(config, model1, None, loader)
                axes[1][0].set_title('Prediction from Rank loss -before')

                axes[1][0].bar([0,1], [sum(np.array(test_result1['output'][0][0])[:int(config.num_bin/2)]), \
                    sum(np.array(test_result1['output'][0][0])[int(config.num_bin/2):])])
                plot(axes[1][0], 2)
                
                test_result = eval(config, model3, None, loader)
                axes[2][0].bar([0,1], [sum(np.array(test_result['output'][0][0])[:int(config.num_bin/2)]), \
                    sum(np.array(test_result['output'][0][0])[int(config.num_bin/2):])])
                axes[2][0].set_title('Prediction from Rank loss -after')
                plot(axes[2][0], 2)

                test_result = eval(config, model2, None, loader)
                axes[1][1].set_title('Prediction from DV loss -before')
                axes[1][1].bar(keys, test_result['output'][0][0])
                plot(axes[1][1], config.num_bin)
            
                test_result = eval(config, model4, None, loader)
                axes[2][1].set_title('Prediction from DV loss -after')
                axes[2][1].bar(keys, test_result['output'][0][0])
                plot(axes[2][1], 2)
                fig.tight_layout()
                plt.savefig(os.path.join(config.save_traj, f'{str(i)}.png'))
                print(os.path.join(config.save_traj, f'{str(i)}.png'))
                plt.clf()
                plt.close()
        else:
            print(i,"pass")
def plot(axes, num_bin):
    axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    axes.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if num_bin==2:
        axes.set_ylim([0, 1])
    if num_bin==50:
        axes.xaxis.set_major_locator(ticker.MultipleLocator(10))
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        axes.set_xlim([0, num_bin-1])
        axes.set_ylim([0, 1])
if __name__ == '__main__':
    main()