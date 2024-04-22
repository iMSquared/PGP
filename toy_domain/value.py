from itertools import accumulate
import os
import time
import random
import numpy as np
import torch as th
from dataclasses import dataclass, replace
from simple_parsing import Serializable
from torch.utils import data

from Learning.model import GPT2, RNN, LSTM, CVAE, ValueNet
from Learning.saver import load_checkpoint
from Learning.utils import CosineAnnealingWarmUpRestarts


@dataclass
class SettingValue(Serializable):
    # Dataset
    path: str = 'toy_domain/Learning/dataset'
    # data_type: str = 'mcts' # 'mcts' or 'success'
    data_type: str = 'success' # 'mcts' or 'success'
    # data_type_1: str = 'success' # 'mcts' or 'success'
    # data_type_2: str = 'mcts' # 'mcts' or 'success'
    randomize: bool = True
    filter: float = 51
    train_file: str = 'sim_3.25/success_mini' # folder name
    neg_file: str = 'sim_3.25/fail_mini'
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
    # cql_loss = 'mse2'
    # cql_loss = 'cql'
    # cql_reg: bool = 0.0
    # cql_logit_activation = False
    # cql_logit_activation = 'tanh'
    # cql_alpha = 1.0
    grad_clip: bool = True
    
    # for value normalization -1 ~ 1
    value_normalization: bool = False
    max_value = 100.0
    min_value = -30.0
    
    # for value distribution
    value_distribution: bool = False
    num_bin: int = 2
    bin_boundary: float = num_bin / 2
    custom_loss: bool = False
    loss_1: bool = False    # CE(V_\theta^+, v^+)
    loss_2: bool = False    # CE(V_\theta^-, v^-)
    loss_3: bool = False    # CE(V_\theta^OOD, v_min)
    loss_4: bool = False    # CE(V_\theta^+, v^_max|v^_min) (cf. after 4/1)
    loss_5: bool = True    # CE(V_\theta^-, v^_max)         (cf. after 4/1)
    loss_6: bool = True    # CE(V_\theta^-, v^_min)         (cf. after 4/1)
    
    alpha_go: bool = False
    rank: bool = False
    preference_loss: bool = True
    preference_softmax: bool = True
    
    q_value: bool = False
    
    test_indicator: bool = False
    # =============================================================================================================================    

    train_pos_en: bool = False
    use_reward: bool = False
    use_mask_padding: bool = True
    coefficient_loss: float = 1e-3

    dropout: float = 0.1
    action_tanh: bool = False

    # Training
    device: str = 'cuda' if th.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
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
    exp_dir: str = '/home/share_folder/exp'
    # exp_dir: str = 'toy_domain/Learning/exp'
    # model_name: str = 'test'
    # model_name: str = '4.17_ValueNet'
    # model_name: str = '3.26_CQL_without_reg'
    # model_name: str = '3.26_CQL_with_reg'
    # model_name: str = '3.27_cql_reg'
    # model_name: str = '3.27_cql_tanh'
    # model_name: str = '3.27_cql_reg_tanh'
    # model_name: str = '3.27_mse'
    # model_name: str = '3.27_mse2'
    # model_name: str = '3.27_mse2_reg'
    # model_name: str = '3.28_mse2_reg'
    # model_name: str = '3.28_mse2_tanh'
    # model_name: str = '3.28_mse2_grad_clip'
    # model_name: str = '3.28_mse2_tanh_grad_clip'
    # model_name: str = '3.28_mse2_lr_1e-4'
    # model_name: str = '3.28_mse2_lr_1e-3'
    # model_name: str = '3.28_mse2_dim128_plus1layer'
    # model_name: str = '3.28_mse2_dim128_plus1layer_tanh'
    # model_name: str = '3.28_mse2_dim128_class16'
    # model_name:str = '3.28_mse2_conti'
    # model_name:str = '3.28_mse2_conti_batch4096'
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
    # model_name: str = '3.30_mse2_dim128_batch4096_dist_bin50_loss1245'
    # model_name: str = '3.30_dreamer_dim128_batch4096_indicator_mse2'
    # model_name: str = '3.30_dreamer_dim128_batch4096_indicator_cql'
    # model_name = '4.1_vd_len30'
    # model_name = '4.1_rank_len30'
    # model_name = '4.1_vd_len4'
    # model_name = '4.1_rank_len4'
    # model_name = '4.2_vd_len30_bigdata'
    # model_name = '4.2_rank_len30_bigdata'
    # model_name = '4.2_vd_len4_bf'
    # model_name = '4.2_rank_len4_bf'
    # model_name = '4.3_vd_len4_bf'
    # model_name = '4.3_rank_len4_bf'
    # model_name = '4.3_rank_len4_bf_scale2'
    # model_name = '4.3_rank_len4_bf_scale0.5'
    # model_name = '4.3_rank_len4_bf_fix'
    # model_name = '4.3_binary_classification_SorF'
    # model_name = f'4.3_preference_sigmoid_steplr_sim'
    # model_name = '4.3_alphago_sim_mini'
    # model_name = '4.4_preference_sigmoid_1'
    # model_name = '4.4_preference_sigmoid_2'
    # model_name = '4.4_preference_softmax_1'
    # model_name = '4.4_preference_softmax_2'
    # model_name = '4.4_preference_sigmoid_1_tiny'
    # model_name = '4.4_preference_sigmoid_2_tiny'
    # model_name = '4.4_preference_softmax_1_tiny'
    # model_name = '4.4_preference_softmax_2_tiny'
    # model_name = '4.4_rank_2'
    # model_name = '4.4_rank_3'
    # model_name = '4.4_rank_tiny_1'
    # model_name = '4.4_rank_tiny_2'
    # model_name = '4.4_rank_tiny_3'
    # model_name = '4.4_rank_mini_1'
    # model_name = '4.4_rank_mini_2'
    # model_name = '4.4_rank_mini_3'
    # model_name = '4.4_alphago_1000_1'
    # model_name = '4.4_alphago_1000_2'
    # model_name = '4.4_alphago_1000_3'
    # model_name = '4.4_preference_softmax_3'
    # model_name = '4.4_preference_sigmoid_3_tiny'
    # model_name = '4.4_preference_softmax_3_tiny'
    # model_name = '4.4_preference_softmax_2_mini'
    # model_name = '4.4_preference_sigmoid_1_mini'
    # model_name = '4.4_alphago_100_1'
    # model_name = '4.4_alphago_100_2'
    # model_name = '4.4_alphago_100_3'
    # model_name = '4.4_alphago_10_1'
    # model_name = '4.4_alphago_10_2'
    # model_name = '4.4_alphago_10_3'
    # model_name = '4.7_alphago_1000_1'
    model_name = '4.7_pref_1000_1'
    # model_name = '4.7_rank_1000_1'
    # model_name = '4.7_rank_100_1'
    # model_name = '4.7_rank_10_1'
    # model_name = '4.7_pref_100_1'
    # model_name = '4.7_pref_10_1'
    
    
    
    
    
    

    print_freq: int = 100 # per train_steps
    train_eval_freq: int = 100 # per train_steps
    test_eval_freq: int = 1 # per epochs
    save_freq: int = 10 # per epochs

    log_para: bool = False
    log_grad: bool = False
    eff_grad: bool = False
    print_num_para: bool = True
    print_in_out: bool = False
    
# class SettingValue(Serializable):
#     # |TODO| modify to automatically change
#     dim_observation: int = 2
#     dim_action: int = 2
#     dim_state: int = 2
#     dim_reward: int = 1
#     max_len: int = 100
#     seq_len: int = 31

#     # Architecture
#     model: str = 'ValueNet'
#     optimizer: str = 'AdamW' # AdamW or AdamWR

#     dim_embed: int = 16
#     dim_hidden: int = 16
#     # dim_embed: int = 128
#     # dim_hidden: int = 128

#     # for GPT
#     dim_head: int = 16
#     num_heads: int = 1
#     dim_ffn: int = 16 * 4
#     num_layers: int = 3
#     # dim_head: int = 128
#     # num_heads: int = 1
#     # dim_ffn: int = 128 * 3
#     # num_layers: int = 3
    
#     # for CVAE
#     latent_size: int = 16
#     dim_condition: int = 16
#     # latent_size: int = 128
#     # dim_condition: int = 128
#     encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
#     decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
#     # encoder_layer_sizes = [dim_embed, latent_size]
#     # decoder_layer_sizes = [latent_size, dim_action]

#     train_pos_en: bool = False
#     use_reward: bool = True
#     use_mask_padding: bool = True
#     randomize: bool = True
#     coefficient_loss: float = 1e-3

#     dropout: float = 0.1
#     action_tanh: bool = False

#     # Training
#     device: str = 'cuda' if th.cuda.is_available() else 'cpu'
#     # device: str = 'cpu'
#     resume: str = 'best.pth' # checkpoint file name for resuming
#     # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
#     epochs: int = 1000

#     # Learning rate
#     # |NOTE| using small learning rate, in order to apply warm up
#     learning_rate: float = 1e-5
#     weight_decay: float = 1e-4
#     warmup_step: int = int(1e3)
#     # For cosine annealing
#     T_0: int = int(1e4)
#     T_mult: int = 1
#     lr_max: float = 0.01
#     lr_mult: float = 0.5

#     # Logging
#     exp_dir: str = 'toy_domain/Learning/exp'
#     model_name: str = '4.17_ValueNet'
#     # model_name: str = '4.25_ValueNet_sim_success_2'

#     print_freq: int = 1000 # per train_steps
#     train_eval_freq: int = 1000 # per train_steps
#     test_eval_freq: int = 10 # per epochs
#     save_freq: int = 1000 # per epochs

#     # Prediction
#     print_in_out: bool = False
#     variance: float = 0.5


class GuideValueNet():
    def __init__(self, config, const):
        self.config = config
        self.const = const

        # model
        self.model_dir = os.path.join(config.exp_dir, config.model_name)
        self.device = th.device(config.device)
        self.model = ValueNet(config).to(self.device)

        self.model.eval()

        # optimizer
        self.optimizer = th.optim.AdamW(self.model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
        
        # learning rate scheduler
        # if config.optimizer == 'AdamW':
        #     self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min((step+1)/config.warmup_step, 1))
        # elif config.optimizer == 'AdamWR':
        #     self.scheduler = CosineAnnealingWarmUpRestarts(
        #         optimizer=self.optimizer,
        #         T_0=config.T_0,
        #         T_mult=config.T_mult,
        #         eta_max=config.lr_max,
        #         T_up=config.warmup_step,
        #         gamma=config.lr_mult
        #     )
        # else:
        #     raise Exception(f'"{config.optimizer}" is not support!! You should select "AdamW" or "AdamWR".')
        self.scheduler = None

        # load checkpoint for resuming
        if config.resume is not None:
            filename = os.path.join(self.model_dir, config.resume)
            if os.path.isfile(filename):
                start_epoch, _, self.model, self.optimizer, self.scheduler = load_checkpoint(config, filename, self.model, self.optimizer, self.scheduler)
                start_epoch += 1
                print("Loaded checkpoint '{}' (epoch {})".format(config.resume, start_epoch))
            else:
                raise Exception("No checkpoint found at '{}'".format(config.resume))

    def sample(self, history, goal):
        """
        infer accumulated reward using neural network
        args:
            history: tuple of current history ((a0, o0, s'0, r0)), (a1, o1, s'1, r1), ... )
        returns:
            pred(float): prediction of accumulated reward
            infer_time(float)
        """
        # fitting form of traj to input of network
        data = self._traj2data(history, goal)

        # predict next action

        with th.no_grad():
            time_start = time.time()
            if self.config.value_distribution:
                pred = self.model.inference(data)
                pred = pred[1].item()
            elif self.config.rank or (self.config.preference_loss and not self.config.preference_softmax):
                pred = self.model.inference_sigmoid(data).detach().item()
            elif self.config.preference_loss and self.config.preference_softmax:
                # pred = th.exp(self.model(data)).item()
                pred = self.model(data).item()
            else:
                pred = self.model(data).item()
            time_end = time.time()
            infer_time = time_end - time_start

        return pred, infer_time
    
    def _traj2data(self, history, goal):
        """
        interface matching for neural network 
        """
        o, a, r, goal_s, timestep, mask = [], [], [], [], [], []
        i = 2

        # get sequences from dataset
        for i, h in enumerate(history):
            a.append(h[0])
            o.append(h[1])
            r.append(h[3])
            timestep.append(i)
        goal_s.append(goal)
        o = np.asarray(o).reshape(1, -1, 2)
        a = np.asarray(a).reshape(1, -1, 2)
        r = np.asarray(r).reshape(1, -1, 1)
        goal_s = np.asarray(goal_s).reshape(1, -1, 2)
        timestep = np.asarray(timestep).reshape(1, -1)

        # padding
        tlen = timestep.shape[1]
        if self.config.randomize:
            mask.append(np.concatenate([np.full((1, self.config.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
        else:
            mask.append(np.concatenate([np.full((1, self.config.seq_len - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 2)), o], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        a = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 2)), a], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        r = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 1)), r], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen)), timestep], axis=1)).to(dtype=th.long, device=th.device(self.config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        goal_s = th.from_numpy(goal_s).to(dtype=th.float32, device=th.device(self.device))

        data = {'observation': o,
            'action': a,
            'reward': r,
            'goal_state': goal_s,
            'timestep': timestep,
            'mask': mask}

        return data


if __name__ == '__main__':
    nn_config = SettingValue()
    guide_poilcy = GuideValueNet(nn_config, [0.5, 0.5])

    history = (((0, 0), (5.296355492413373, -0.24048282626180972), (2.0623819119018396, 2.1958711763514596), 0), ((2.796929254158842, 2.0435689830185737), (4.690556164130332, 4.491229440722575), (4.747127206199204, 4.527105117508303), -1), )
    goal = (0, 0)
    accumulated_reward, inference_time = guide_poilcy.sample(history, goal)
    print(accumulated_reward, type(accumulated_reward))
    print(inference_time)