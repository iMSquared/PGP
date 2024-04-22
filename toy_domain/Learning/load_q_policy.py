import os
import glob
import pickle
import numpy as np
import torch as th
import torch.nn.functional as F
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from dataclasses import dataclass, replace
from simple_parsing import Serializable

import random
import tqdm

# from wandb import set_trace

class LightDarkDataset(Dataset):
    """
    Get a train/test dataset according to the specified settings.
    """
    def __init__(self, config, dataset: List, transform=None, per_node=False):
        self.config = config
        self.seq_len = config.seq_len
        self.dataset = dataset # list of data file name
        self.transform = transform
        self.p_sample_list = []
        self.per_node = per_node
        for i in tqdm.tqdm(dataset):
            with open(i, 'rb') as f:
                try:
                    d = pickle.load(f)
                except pickle.UnpicklingError:
                    pass
                # idx_tree_node = [d[2], d[3]]
                # self.p_sample_list.append(d[3]-d[2])
                self.p_sample_list.append(len(d[0]))

    def __len__(self):
        # return np.sum(np.asarray(self.p_sample_list))
        # return np.sum(len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, index):
        d = self.dataset[index]
        with open(d, 'rb') as f:
            try:
                # print(self.dataset[index])
                d = pickle.load(f)
            except pickle.UnpicklingError:
                pass
        
        if self.transform:
            d = self.transform(d)

        if self.config.randomize:
            traj = d[0]
            goal_state = d[1]
        else:
            traj = d
        
        # print('========================================')
        # print(traj)
        data_action = traj[:, 0].tolist()
        data_observation = traj[:, 1].tolist()
        data_next_state = traj[:, 2].tolist()
        data_reward = traj[:, 3].tolist()
        # idx_tree_node = [d[2], d[3]]


        if self.per_node:
            # placeholder for data
            action = np.zeros((self.seq_len, self.config.dim_action))
            observation = np.zeros((self.seq_len, self.config.dim_observation))
            next_state = np.zeros((self.seq_len, self.config.dim_state))
            reward = np.zeros((self.seq_len, self.config.dim_reward))

            traj_idx = random.randint(1, len(traj))
            action[:traj_idx] = np.array(data_action)[:traj_idx]
            observation[:traj_idx] = np.array(data_observation)[:traj_idx]
            next_state[:traj_idx] = np.array(data_next_state)[:traj_idx]
            reward[:traj_idx] = data_reward[-1]
            mask = np.zeros((self.seq_len,))
            mask[traj_idx+1:] = 1
        else:
            traj_idx = len(traj)
            action = data_action
            observation = data_observation
            next_state = data_next_state
            reward = data_reward
            mask = None


        data = {'action': action,
                'observation': observation,
                'next_state': next_state,
                'reward': reward}
                # 'idx_tree_node': idx_tree_node}
        
        if self.config.randomize:
            data['goal_state'] = np.asarray(goal_state)

        return data


class BatchMakerQPolicy():
    def __init__(self, config, value_model):
        self.config = config
        self.device = config.device
        self.value_model = value_model
        
    def _next_action(self, history, r_range=(0,2), theta_range=(0, 2*np.pi), num_sample=1):
        r = np.random.uniform(r_range[0], r_range[1], num_sample)
        theta = np.random.uniform(theta_range[0], theta_range[1], num_sample)
        _action_x = r * np.cos(theta)
        _action_y = r * np.sin(theta)
        _action = np.concatenate([_action_x.reshape(-1, 1),_action_y.reshape(-1, 1)], axis=1)
        return _action

    def __call__(self, data):
        o, a, next_a, goal_s, w, timestep, mask = [], [], [], [], [], [], []
        for traj in data:
            if len(traj['observation']) > 2:
                i = np.random.randint(1, len(traj['observation']) - 1)
            else:
                i = len(traj['observation']) -1
            # if 'traj_len' in traj.keys():
            #     i = traj['traj_len']-1
            # elif self.config.use_sampler:
            #     i = np.random.randint(low=traj['idx_tree_node'][0]+1, high=traj['idx_tree_node'][1]+1, size=1).item()-1
            # else:
            #     if len(traj['observation']) == 2:
            #         i = 1
            #     else:
            #         i = np.random.randint(1, len(traj['observation']) - 1)

            # get sequences from dataset
            tlen = i
            sampled_o = np.asarray(traj['observation'])[:i].reshape(1, -1, 2)
            sampled_o = np.concatenate([np.zeros((1, 31 - tlen, 2)), sampled_o], axis=1)
            sampled_o = np.tile(sampled_o, (self.config.num_candidate, 1, 1))
            o.append(sampled_o)
            
            sampled_a = np.asarray(traj['action'])[:i].reshape(1, -1, 2)
            sampled_a = np.concatenate([np.zeros((1, 31 - tlen, 2)), sampled_a], axis=1)
            sampled_a = np.tile(sampled_a, (self.config.num_candidate, 1, 1))
            a.append(sampled_a)
            
            if self.config.randomize:
                sampled_goal_s = np.asarray(traj['goal_state']).reshape(1, -1, 2)
                sampled_goal_s = np.tile(sampled_goal_s, (self.config.num_candidate, 1, 1))
                goal_s.append(sampled_goal_s)
                
            sampled_timestep = np.arange(0, i).reshape(1, -1)
            sampled_timestep = np.concatenate([np.zeros((1, 31 - tlen)), sampled_timestep], axis=1)
            sampled_timestep = np.tile(sampled_timestep, (self.config.num_candidate, 1, 1))
            timestep.append(sampled_timestep)

            if self.config.randomize:
                sampled_mask = np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1)
                sampled_mask = np.tile(sampled_mask, (self.config.num_candidate, 1))
                mask.append(sampled_mask)
            else:
                sampled_mask = np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1)
                sampled_mask = np.tile(sampled_mask, (self.config.num_candidate, 1))
                mask.append(sampled_mask)        
            
            # get importance weight
            sampled_a_q = np.asarray(traj['action'][1:i]).reshape(1, -1, 2)
            sampled_a_q = np.concatenate([np.zeros((1, 31 - tlen, 2)), sampled_a_q], axis=1)
            sampled_a_q = np.tile(sampled_a_q, (self.config.num_candidate, 1, 1))
            next_action_candidates = self._next_action(None, num_sample=self.config.num_candidate).reshape(-1, 1, 2)
            next_a.append(next_action_candidates)
            sampled_a_q = np.concatenate([sampled_a_q, next_action_candidates], axis=1)
            
            input_value_model_sampled_o = th.from_numpy(sampled_o).to(dtype=th.float32, device=th.device(self.device))
            input_value_model_sampled_a_q = th.from_numpy(sampled_a_q).to(dtype=th.float32, device=th.device(self.device))
            input_value_model_sampled_timestep = th.from_numpy(sampled_timestep).to(dtype=th.long, device=th.device(self.device))
            input_value_model_sampled_mask = th.from_numpy(sampled_mask).to(device=th.device(self.device))
            if self.config.randomize:
                input_value_model_sampled_goal_s = th.from_numpy(sampled_goal_s).to(dtype=th.float32, device=th.device(self.config.device))
            input_value_model = {
                'observation': input_value_model_sampled_o,
                'action': input_value_model_sampled_a_q,
                'timestep': input_value_model_sampled_timestep,
                'mask': input_value_model_sampled_mask,
            }
            if self.config.randomize:
                input_value_model['goal_state'] = input_value_model_sampled_goal_s
            
            # predict the value of sampled next actions
            with th.no_grad():
                pred = self.value_model(input_value_model).squeeze()
                prob = th.nn.functional.softmax(pred, dim=0)
            w.append(prob)
            

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        # r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        # next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        # next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        # value = th.from_numpy(np.concatenate(accumulated_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        # accumulated_r = th.from_numpy(np.concatenate(accumulated_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        # success_or_fail = th.where(accumulated_r > 0, 1.0, 0.0).to(dtype=th.float32, device=th.device(self.config.device))
        if self.config.randomize:
            goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        # timestep_idx = th.Tensor(timestep_idx).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        importance_weight = th.cat(w, dim=0).to(dtype=th.float32, device=th.device(self.device))

        
        out = {'observation': o,
            'action': a,
            # 'reward': r,
            'next_action': next_a,
            'importance_weight': importance_weight,
            # 'next_reward': next_r,
            # 'next_state': next_s,
            # 'accumulated_reward': accumulated_r,
            # 'success_or_fail': success_or_fail,
            # 'value': value,
            'timestep': timestep,
            # 'timestep_idx': timestep_idx,
            'mask': mask}
        
        if self.config.randomize:
            out['goal_state'] = goal_s

        return out


def get_loader(config, dataset: Dict,
               value_model,
               transform=None, collate_fn=None, train=True):
    dataset = LightDarkDataset(config, dataset, transform)
    batcher = BatchMakerQPolicy(config, value_model)

    if config.use_sampler:
        sampler = WeightedRandomSampler(dataset.p_sample_list, config.batch_size)
    else:
        sampler = None

    loader = DataLoader(dataset,
                        batch_size=config.batch_size,
                        shuffle=config.shuffle,
                        sampler=sampler,
                        collate_fn=batcher)
    return loader


if __name__ == '__main__':
    @dataclass
    class Settings(Serializable):
        # Dataset
        path: str = 'toy_domain/Learning/dataset'
        data_type: str = 'success' # 'mcts' or 'success'
        file: str = 'sim_3.25/fail_mini' # folder name - mcts / file name - success traj.
        batch_size: int = 4 # 100steps/epoch
        shuffle: bool = True # for using Sampler, it should be False
        use_sampler: bool = False
        max_len: int = 100
        seq_len: int = 31
        randomize: bool = True

        # |TODO| modify to automatically change
        dim_observation: int = 2
        dim_action: int = 2
        dim_state: int = 2
        dim_reward: int = 1

        # Architecture
        model: str = 'CVAE' # GPT or RNN or LSTM or CVAE
        optimizer: str = 'AdamW' # AdamW or AdamWR

        dim_embed: int = 8
        dim_hidden: int = 8

        # for GPT
        dim_head: int = 8
        num_heads: int = 1
        dim_ffn: int = 8 * 4
        num_layers: int = 3

        # for CVAE
        latent_size: int = 32
        dim_condition: int = 32
        # encoder_layer_sizes = [dim_embed, dim_embed + dim_condition, latent_size]
        # decoder_layer_sizes = [latent_size, latent_size + dim_condition, dim_action]
        encoder_layer_sizes = [dim_embed, latent_size]
        decoder_layer_sizes = [latent_size, dim_action]

        train_pos_en: bool = False
        use_reward: bool = True
        use_mask_padding: bool = True
        coefficient_loss: float = 1e-3

        dropout: float = 0.1
        action_tanh: bool = False

        # Training
        device: str = 'cuda' if th.cuda.is_available() else 'cpu'
        resume: str = None # checkpoint file name for resuming
        pre_trained: str = None # checkpoint file name for pre-trained model
        # |NOTE| Large # of epochs by default, Such that the tranining would *generally* terminate due to `train_steps`.
        epochs: int = 1000

        # Learning rate
        # |NOTE| using small learning rate, in order to apply warm up
        learning_rate: float = 1e-5
        weight_decay: float = 1e-4
        warmup_step: int = int(1e3)
        # For cosine annealing
        T_0: int = int(1e4)
        T_mult: int = 1
        lr_max: float = 0.01
        lr_mult: float = 0.5

        # Logging
        exp_dir: str = 'Learning/exp'
        model_name: str = 'test'
        print_freq: int = 1000 # per train_steps
        train_eval_freq: int = 1000 # per train_steps
        test_eval_freq: int = 10 # per epochs
        save_freq: int = 100 # per epochs

        log_para: bool = False
        log_grad: bool = False
        eff_grad: bool = False
        print_num_para: bool = True
        print_in_out: bool = False


    config = Settings()
    dataset_path = os.path.join(os.getcwd(), 'toy_domain/Learning/dataset')
    # dataset_filename = 'sim_success_exp_const_30_std0.5_randomize_1'
    dataset_filename = config.file

    # with open(os.path.join(dataset_path, dataset_filename), 'rb') as f:
    #     dataset = pickle.load(f)
    # print('#trajectories of test_dataset:', len(dataset['observation']))

    dataset = glob.glob(f'{dataset_path}/{dataset_filename}/*.pickle')
    print('#trajectories of train_dataset:', len(dataset))

    data_loader = get_loader(config, dataset)
    # data_loader = get_loader_multi_target(config, dataset)

    import shutil
    dst_dir = '/home/jiyong/workspace/POMDP/toy_domain/Learning/dataset/sim_3.25/out'
    for i, data in enumerate(data_loader):
        print(i)
        print(dataset[i])
        filt = data['accumulated_reward'].item() > 50
        print(filt)
        if filt:
            shutil.move(dataset[i], dst_dir)