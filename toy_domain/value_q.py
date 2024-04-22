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
class SettingQValue():
    def __init__(self):
        # Dataset
        self.randomize: bool = True
        
        self.max_len: int = 100
        self.seq_len: int = 31

        self.dim_observation: int = 2
        self.dim_action: int = 2
        self.dim_state: int = 2
        self.dim_reward: int = 1

        # Architecture
        self.model: str = 'ValueNet' # GPT or RNN or LSTM or CVAE or ValueNet or PolicyValueNet, or ValueNetDiscreteRepresentation
        
        self.optimizer: str = 'AdamW' # AdamW or AdamWR

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
        
        self.value_distribution: bool = False
        self.test_indicator: bool = False
        
        self.train_pos_en: bool = False
        self.use_reward: bool = False
        self.use_mask_padding: bool = True
        self.coefficient_loss: float = 1e-3

        self.dropout: float = 0.1
        self.action_tanh: bool = False
        
        self.learning_rate: float = 1e-4
        self.weight_decay: float = 1e-4
        
        self.device: str = 'cuda' if th.cuda.is_available() else 'cpu'
        # self.device: str = 'cpu'
        self.exp_dir: str = '/home/share_folder/exp'
        self.resume: str = 'best.pth' # checkpoint file name for resuming
        # self.resume: str = 'ckpt_epoch_7500.pth'
        
        # =============================================================================================================================    
        
        # if config['model_type'] == 'preference':
        #     self.alpha_go: bool = False
        #     self.rank: bool = False
        #     self.preference_loss: bool = True
        #     self.preference_softmax: bool = True
        # elif config['model_type'] == 'rank':
        #     self.alpha_go: bool = False
        #     self.rank: bool = True
        #     self.preference_loss: bool = False
        #     self.preference_softmax: bool = False
        # elif config['model_type'] == 'alphago':
        #     self.alpha_go: bool = True
        #     self.rank: bool = False
        #     self.preference_loss: bool = False
        #     self.preference_softmax: bool = False
            
        self.alpha_go: bool = False
        self.rank: bool = False
        self.preference_loss: bool = True
        self.preference_softmax: bool = True

        self.q_value: bool = True
        self.batch_size: int = 1

        # self.model_name = config['model_name']
        self.model_name = '4.10_q_pref_1000_lr1e-4_1'
        # =============================================================================================================================    


class GuideQValueNet():
    def __init__(self, config):
        self.config = config

        # model
        self.model_dir = os.path.join(config.exp_dir, config.model_name)
        self.device = th.device(config.device)
        self.model = ValueNet(config).to(self.device)

        self.model.eval()

        # optimizer
        self.optimizer = th.optim.AdamW(self.model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
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

    def sample(self, history, goal, action):
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
        for k, v in data.items():
            if k == 'observation' or k == 'action' or k == 'goal_state':
                data[k] = v.expand(self.config.batch_size, -1, -1)
            elif k == 'timestep' or k == 'mask':
                data[k] = v.expand(self.config.batch_size, -1)
        sampled_action = np.asarray(action)
        sampled_action_tensor = th.from_numpy(sampled_action).to(dtype=th.float32, device=th.device(self.device)).reshape(1, -1, 2)
        data['action'] = th.cat((data['action'], sampled_action_tensor), dim=1)

        # predict the value of sampled next actions
        with th.no_grad():
            time_start = time.time()
            # pred = th.exp(self.model(data)).item()
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
            if i != 0:
                a.append(h[0])
            o.append(h[1])
            # r.append(h[3])
            timestep.append(i)
        goal_s.append(goal)
        o = np.asarray(o).reshape(1, -1, 2)
        a = np.asarray(a).reshape(1, -1, 2)
        # r = np.asarray(r).reshape(1, -1, 1)
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
        # r = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen, 1)), r], axis=1)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate([np.zeros((1, 31 - tlen)), timestep], axis=1)).to(dtype=th.long, device=th.device(self.config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        goal_s = th.from_numpy(goal_s).to(dtype=th.float32, device=th.device(self.device))

        data = {'observation': o,
            'action': a,
            # 'reward': r,
            'goal_state': goal_s,
            'timestep': timestep,
            'mask': mask}

        return data


if __name__ == '__main__':
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--model_name", type=str, default="4.10_q_pref_1000_lr1e-4_1")
    params = parser.parse_args()
    model_name = params.model_name
    config_path = os.path.join("/home/share_folder/exp", model_name, "config_planner.yaml")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    nn_config = SettingQValue(config)
    guide_poilcy = GuideQValueNet(nn_config)

    history = (((0, 0), (5.296355492413373, -0.24048282626180972), (2.0623819119018396, 2.1958711763514596), 0), ((2.796929254158842, 2.0435689830185737), (4.690556164130332, 4.491229440722575), (4.747127206199204, 4.527105117508303), -1), )
    goal = (0, 0)
    next_action, next_action_value, inference_time = guide_poilcy.sample(history, goal)
    print(next_action, type(next_action))
    print(next_action_value, type(next_action_value))
    print(inference_time)