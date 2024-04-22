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
# import wandb

from load import get_loader
from model import GPT2, RNN, LSTM, CVAE, ValueNet, ValueNetDiscreteRepresentation
from loss import RegressionLossPolicy, RegressionLossValue, ELBOLoss, CQLLoss, RegressionLossValueWithNegativeData
from trainer import Trainer
from evaluator import Evaluator
from saver import save_checkpoint, load_checkpoint
from utils import ModelAsTuple, CosineAnnealingWarmUpRestarts, log_gradients

import matplotlib.pyplot as plt


class Settings(Serializable):
    # Dataset
    path: str = 'toy_domain/Learning/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 0
    test_file = ['sim_3.25/success_mini', 'sim_3.25/fail_mini']
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
    cql_loss = 'mse'
    # cql_loss = 'mse2'
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
    # pre_trained: str = '4.17_CVAE/best.pth'
    # pre_trained: str = '4.17_ValueNet/best.pth'
    # pre_trained: str = '3.11_CVAE_mcts_1_dim16/best.pth'
    # pre_trained: str = '3.5_CVAE_sim_mcts_2_dim16/best.pth'
    # pre_trained: str = '2.27_CVAE_sim_mcts_1_dim16/best.pth'
    # pre_trained: str = '2.8_CVAE_sim_dim16/best.pth'
    # pre_trained: str = '12.7_CVAE_mcts2/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '11.23_CVAE_randomized/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '11.29_CVAE_mcts1_filtered/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '12.27_CVAE_sim_huge/best.pth' # checkpoint file name for pre-trained model
    # pre_trained: str = '12.27_CVAE_sim_huge_x/best.pth' # checkpoint file name for pre-trained model
    # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
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
    exp_dir: str = 'toy_domain/Learning/exp'
    # model_name: str = '3.28_mse2_dim128_batch4096'
    # model_name: str = '3.28_dreamer_dim128_batch4096'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm'
    # model_name: str = '3.28_mse2_dim128_batch4096_tanh_value_norm'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm_reg1'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_reg'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_bigdata'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse_bigdata'
    # ============================================================
    # model_name: str = '3.28_mse2_dim128_batch4096'
    # model_name: str = '3.28_dreamer_batch4096'
    # model_name: str = '3.28_mse2_dim128_batch4096_bigdata'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_bigdata'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm_bigdata'
    # model_name: str = '3.28_mse2_dim128_batch4096_tanh_value_norm'
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_value_norm_reg'    
    # model_name: str = '3.28_dreamer_dim128_batch4096_tanh_reg'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_bigdata'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse'
    # model_name: str = '3.29_mse2_dim128_batch4096_dist_eval_mse_bigdata_re'
    # model_name: str = '3.30_mse2_dim128_batch4096_dist_eval_mse_bin50'
    # model_name: str = '3.30_mse2_dim128_batch4096_dist_bin50_loss45'
    model_name: str = '3.30_mse2_dim128_batch4096_dist_bin50_loss1245'
    # model_name: str = '3.30_dreamer_dim128_batch4096_indicator_mse2'
    # model_name: str = '3.30_dreamer_dim128_batch4096_indicator_cql'
    

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    

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
                test_result['output'].append(pred[1].detach().item())
            else:
                test_result['target'] += [target['accumulated_reward'].detach()]
                test_result['output'] += [pred.squeeze().detach()]
        
    return test_result


def main():
    config = Settings()
    dataset_filename = config.test_file
    dataset_path = os.path.join(os.getcwd(), config.path)
    
    if not os.path.exists(config.exp_dir):
        os.mkdir(config.exp_dir)
    model_dir = os.path.join(config.exp_dir, config.model_name)

    # model
    device = th.device(config.device)
    if config.model == 'ValueNet':
        model = ValueNet(config).to(device)
    elif config.model == 'ValueNetDiscreteRepresentation':
        model = ValueNetDiscreteRepresentation(config).to(device)
    else:
        raise Exception(f'"{config.model}" is not support!! You should select "GPT", "RNN", or "LSTM".')

    # optimizer
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    
    # learning rate scheduler
    # if config.optimizer == 'AdamW':
    #     scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/config.warmup_step, 1))
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
    scheduler = None    

    # load checkpoint for resuming
    filename = os.path.join(model_dir, config.resume)
    if os.path.isfile(filename):
        epoch, best_error, model, optimizer, scheduler = load_checkpoint(config, filename, model, optimizer, scheduler)
        print("Loaded checkpoint '{}' (epoch {})".format(config.resume, epoch))
    else:
        raise Exception("No checkpoint found at '{}'".format(config.resume))
    print(f'===== Evaluate {epoch} epoch =====')
    
    
    for dataset_filename in config.test_file:
        
        dataset = glob.glob(f'{dataset_path}/{dataset_filename}/*.pickle')
        dataset = dataset[-1000:]
        
        # if config.filter:
        #     filtered_data = []
        #     for data in dataset:
        #         with open(data, 'rb') as f:
        #             traj = pickle.load(f)
        #             if traj[-1] < config.filter:
        #                 filtered_data.append(data)

        #     dataset = filtered_data
                
        data_loader = get_loader(config, dataset)
        print('#trajectories of dataset:', len(data_loader))
        
        test_result = eval(config, model, dataset, data_loader)
            
        # Logging
        plt.boxplot([np.asarray(test_result['output']), np.asarray(test_result['target'])], notch=True, whis=2.5)
        plt.ylabel('value')
        plt.xticks([1, 2], ['output', 'target'])
        if config.value_normalization:
            plt.ylim([-1, 1])
        else:
            plt.ylim([-30, 100])
        fig_name = os.path.split(dataset_filename)[-1]
        plt.savefig(os.path.join(model_dir, f'{fig_name}.png'))
        plt.clf()
        
    return


if __name__ == '__main__':
    main()