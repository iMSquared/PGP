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

    # def __init__(self, config, dataset: Dict, transform=None):
    #     self.config = config
    #     self.dataset = dataset
    #     self.transform = transform
        
    #     # for get_batch()
    #     self.device = config.device
    #     self.max_len = config.max_len
    #     self.seq_len = config.seq_len
    #     self.dim_observation = config.dim_observation
    #     self.dim_action = config.dim_action
    #     self.dim_state = config.dim_state
    #     self.dim_reward = config.dim_reward

    #     # # for WeightedRandomSampler
    #     # self.p_sample = dataset['p_sample']

    # def __len__(self):
    #     return len(self.dataset['observation'])

    # def __getitem__(self, index):
    #     observation = self.dataset['observation'][index]
    #     action = self.dataset['action'][index]
    #     reward = self.dataset['reward'][index]
    #     next_state = self.dataset['next_state'][index]
    #     traj_len = self.dataset['traj_len'][index]
    #     goal_state = self.dataset['goal_state'][index]
    #     total_reward = self.dataset['total_reward'][index]

    #     sample = {'observation': observation,
    #               'action': action,
    #               'reward': reward,
    #               'next_state': next_state,
    #               'goal_state': goal_state,
    #               'total_reward': total_reward,
    #               'traj_len': traj_len}
        
    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample


class BatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def __call__(self, data):
        o, a, r, next_a, next_s, next_r, goal_s, accumulated_r, timestep, mask, timestep_idx = [], [], [], [], [], [], [], [], [], [], []
        if self.config.test_indicator:
            indicator = []
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
            o.append(np.asarray(traj['observation'])[:i].reshape(1, -1, 2))
            a.append(np.asarray(traj['action'][:i]).reshape(1, -1, 2))
            r.append(np.asarray(traj['reward'][:i]).reshape(1, -1, 1))
            next_a.append(np.asarray(traj['action'])[i].reshape(1, -1, 2))
            next_r.append(np.asarray(traj['reward'])[i].reshape(1, -1, 1))
            next_s.append(np.asarray(traj['next_state'])[1:i+1].reshape(1, -1, 2))
            accumulated_r.append(np.sum(np.asarray(traj['reward'][i:])).reshape(1, -1))
            
            if self.config.test_indicator:
                if accumulated_r[-1] > 50:
                    indicator.append(np.asarray([1.]).reshape(1, -1, 1))
                else:
                    indicator.append(np.asarray([0.]).reshape(1, -1, 1))

            # if np.sum(np.asarray(traj['reward'][i:])) < 0.0:
            #     pdb.set_trace()

            if self.config.randomize:
                goal_s.append(traj['goal_state'].reshape(1, -1, 2))
            timestep.append(np.arange(0, i).reshape(1, -1))
            timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

            # padding
            # |FIXME| check padded value & need normalization?
            tlen = o[-1].shape[1]
            o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), a[-1]], axis=1)
            # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
            next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
            timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
            timestep_idx.append(i)
            if self.config.randomize:
                mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
            else:
                mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

            if self.config.test_indicator:
                indicator[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), np.full((1, tlen, 1), indicator[-1])], axis=1)

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        value = th.from_numpy(np.concatenate(accumulated_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        accumulated_r = th.from_numpy(np.concatenate(accumulated_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        success_or_fail = th.where(accumulated_r > 0, 1.0, 0.0).to(dtype=th.float32, device=th.device(self.config.device))
        if self.config.randomize:
            goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        timestep_idx = th.Tensor(timestep_idx).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
        if self.config.test_indicator:
            indicator = th.from_numpy(np.concatenate(indicator, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        
        # value distribution or value normalization to -1 ~ 1
        if self.config.value_distribution:            
            accumulated_r = th.floor((accumulated_r - self.config.min_value) * (self.config.num_bin - 1) / (self.config.max_value - self.config.min_value))
            accumulated_r = accumulated_r.to(dtype=th.long)
            for bin in accumulated_r:
                if bin.item() >= (self.config.num_bin) or bin.item() < 0:
                    print(bin.item())
                    
            # no need to make one hot for nn.CrossEntropyLoss
            # accumulated_r_bin_one_hot = F.one_hot(accumulated_r_bin.to(th.int64), num_classes=self.config.num_bin)
            # accumulated_r = accumulated_r_bin_one_hot
        
        elif self.config.value_normalization:
            median = (self.config.max_value + self.config.min_value) * 0.5
            interval = self.config.max_value - self.config.min_value
            accumulated_r = (accumulated_r - median) / (interval * 0.5)
        
        out = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'accumulated_reward': accumulated_r,
            'success_or_fail': success_or_fail,
            'value': value,
            'timestep': timestep,
            'timestep_idx': timestep_idx,
            'mask': mask}
        
        if self.config.randomize:
            out['goal_state'] = goal_s
            
        if self.config.test_indicator:
            out['indicator'] = indicator

        return out


class MultiTargetLightDarkDataset(Dataset):
    def __init__(self, config, dataset: Dict, transform=None):
        self.config = config
        self.dataset = dataset
        self.transform = transform
        
        # for get_batch()
        self.device = config.device
        self.max_len = config.max_len
        self.seq_len = config.seq_len
        self.dim_observation = config.dim_observation
        self.dim_action = config.dim_action
        self.dim_state = config.dim_state
        self.dim_reward = config.dim_reward

        # # for WeightedRandomSampler
        # self.p_sample = dataset['p_sample']

    def __len__(self):
        return len(self.dataset['observation'])

    def __getitem__(self, index):
        # get sequences from dataset
        observation = self.dataset['observation'][index]
        action = self.dataset['action'][index]
        reward = self.dataset['reward'][index]
        next_state = self.dataset['next_state'][index]
        traj_len = self.dataset['traj_len'][index]
        goal_state = self.dataset['goal_state'][index]
        total_reward = self.dataset['total_reward'][index]

        traj = {'observation': observation,
                  'action': action,
                  'reward': reward,
                  'next_state': next_state,
                  'traj_len': traj_len,
                  'goal_state': goal_state,
                  'total_reward': total_reward}

        if len(traj['observation']) == 2:
            i = 1
        else:
            i = np.random.randint(1, len(traj['observation']) - 1)

        sample = self._collect_target(traj, i)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _collect_target(self, traj, i):
        # truncate & fit interface of sample to model
        o, a, r, next_a, next_s, next_r, goal_s, total_r, timestep, mask = [], [], [], [], [], [], [], [], [], []
        o.append(traj['observation'][:i].reshape(-1, 2))
        a.append(traj['action'][:i].reshape(-1, 2))
        r.append(traj['reward'][:i].reshape(-1, 1))
        next_a.append(np.round(traj['action'][i], 4).reshape(-1, 2))
        next_r.append(traj['reward'][i].reshape(-1, 1))
        next_s.append(traj['next_state'][1:i+1].reshape(-1, 2))
        total_r.append(traj['total_reward'])
        goal_s.append(traj['goal_state'].reshape(1, -1, 2))
        timestep.append(np.arange(0, i))
        timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

        # collect multi-target indices
        # |TODO| how to use full len?
        target_index = []
        if i == 1:
            o0 = np.round(traj['observation'][0], 4)
            for idx in range(len(self.dataset['observation'])):
                if np.array_equal(np.round(self.dataset['observation'][idx][0] ,4), o0):
                    target_index.append(idx)

        elif i == 2:
            # take first action in sample
            a1 = np.round(traj['action'][1], 4)
            o1 = np.round(traj['observation'][1], 4)
            for idx in range(len(self.dataset['observation'])):
                if len(self.dataset['action'][idx]) < i+1:
                    continue
                if np.array_equal(np.round(self.dataset['action'][idx][1], 4), a1) and np.array_equal(np.round(self.dataset['observation'][idx][1], 4), o1):
                    target_index.append(idx)
        
        elif i == 3:
            a1 = np.round(traj['action'][1], 4)
            o1 = np.round(traj['observation'][1], 4)
            a2 = np.round(traj['action'][2], 4)
            o2 = np.round(traj['observation'][2], 4)
            for idx in range(len(self.dataset['observation'])):
                if len(self.dataset['action'][idx]) < i+1:
                    continue
                if np.array_equal(np.round(self.dataset['action'][idx][1], 4), a1) and np.array_equal(np.round(self.dataset['observation'][idx][1], 4), o1) and np.array_equal(np.round(self.dataset['action'][idx][2], 4), a2) and np.array_equal(np.round(self.dataset['observation'][idx][2], 4), o2):
                    target_index.append(idx)

        # Collect multi-targets
        if target_index:
            for t in target_index:
                # |FIXME| IndexError: index 3 is out of bounds for axis 0 with size 3
                if len(self.dataset['action'][t]) < i+1:
                    continue
                next_a.append(np.round(self.dataset['action'][t][i], 4).reshape(-1, 2))

        # padding
        tlen = o[-1].shape[-2]
        o[-1] = np.concatenate([np.zeros((31 - tlen, 2)), o[-1]], axis=-2)
        a[-1] = np.concatenate([np.zeros((31 - tlen, 2)), a[-1]], axis=-2)
        r[-1] = np.concatenate([np.zeros((31 - tlen, 1)), r[-1]], axis=-2)
        next_s[-1] = np.concatenate([np.zeros((31 - tlen, 2)), next_s[-1]], axis=-2)
        timestep[-1] = np.concatenate([np.zeros((31 - tlen)), timestep[-1]])
        # mask.append(np.concatenate([np.full(31 - tlen, False, dtype=bool), np.full(tlen, True, dtype=bool)]))
        mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        # next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        total_r = th.from_numpy(np.asarray(total_r).reshape(-1, 1)).to(dtype=th.float32, device=th.device(self.config.device))
        goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.config.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.config.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.config.device))

        next_a = np.array(next_a).reshape(-1, 2)
        next_a = np.unique(next_a, axis=-2)
        next_a = th.from_numpy(next_a).to(dtype=th.float32, device=th.device(self.config.device))
        
        data = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_reward': next_r,
            'next_state': next_s,
            'goal_state': goal_s,
            'total_reward': total_r,
            'timestep': timestep,
            'mask': mask}

        return data

# # |TODO| make collate_fn for multi-target
# class MultiTargetBatchMaker():
#     def __init__(self, config):
#         self.config = config
#         self.device = config.device

#     def __call__(self, data):
#         o, a, r, next_a, next_s, next_r, timestep, mask = [], [], [], [], [], [], [], []
#         for traj in data:
#             if len(traj['observation']) == 2:
#                 i = 1
#             else:
#                 i = np.random.randint(1, len(traj['observation']) - 1)
            
#             # get sequences from dataset
#             o.append(traj['observation'][:i].reshape(1, -1, 2))
#             a.append(traj['action'][:i].reshape(1, -1, 2))
#             r.append(traj['reward'][:i].reshape(1, -1, 1))
#             next_a.append(traj['action'][i].reshape(1, -1, 2))
#             next_r.append(traj['reward'][i].reshape(1, -1, 1))
#             next_s.append(traj['next_state'][1:i+1].reshape(1, -1, 2))
#             timestep.append(np.arange(0, i).reshape(1, -1))
#             timestep[-1][timestep[-1] >= 31] = 31 - 1  # padding cutoff

#             # padding
#             # |FIXME| check padded value & need normalization?
#             tlen = o[-1].shape[1]
#             o[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), o[-1]], axis=1)
#             a[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), a[-1]], axis=1)
#             # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
#             r[-1] = np.concatenate([np.zeros((1, 31 - tlen, 1)), r[-1]], axis=1)
#             next_s[-1] = np.concatenate([np.zeros((1, 31 - tlen, 2)), next_s[-1]], axis=1)
#             timestep[-1] = np.concatenate([np.zeros((1, 31 - tlen)), timestep[-1]], axis=1)
#             mask.append(np.concatenate([np.full((1, 31 - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

#         o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_r = th.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
#         timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
#         mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        
#         out = {'observation': o,
#             'action': a,
#             'reward': r,
#             'next_action': next_a,
#             'next_reward': next_r,
#             'next_state': next_s,
#             'timestep': timestep,
#             'mask': mask}

#         return out


class MCTSLightDarkDataset(Dataset):
    """
    Get a train/test dataset according to the specified settings.
    """
    def __init__(self, config, dataset: List, transform=None):
        self.config = config
        self.dataset = dataset # list of data file name
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        traj = self.dataset[index]
        with open(traj, 'rb') as f:
            traj = pickle.load(f)
        
        if self.transform:
            data = self.transform(traj)
        
        idx = np.random.choice(len(traj) - 2)
        sample = traj[idx]
        
        history = np.asarray(sample[0])
        actions = sample[1]
        p_action = sample[2]
        num_visit_action = sample[3]
        val_node = sample[5]
        if self.config.randomize:
            goal_state = traj[-2]
        total_reward = traj[-1]

        i = np.random.choice(len(actions), p=p_action)
        sampled_next_action = actions[i]

        data = {'action': history[:, 0].tolist(),
                'observation': history[:, 1].tolist(),
                'next_state': history[:, 2].tolist(),
                'reward': history[:, 3].tolist(),
                'next_action': sampled_next_action,
                'total_reward': total_reward}
        
        if self.config.randomize:
            data['goal_state'] = goal_state

        return data


class MCTSBatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dim_action = config.dim_action
        self.dim_observation = config.dim_observation
        self.dim_state = config.dim_state
        self.dim_reward = config.dim_reward
        self.seq_len = config.seq_len


    def __call__(self, data):
        a, o, r, next_a, next_s, goal_s, timestep, mask = [], [], [], [], [], [], [], []
        for d in data:
            # get sequences from dataset
            a.append(np.asarray(d['action']).reshape(1, -1, self.dim_action))
            o.append(np.asarray(d['observation']).reshape(1, -1, self.dim_observation))
            r.append(np.asarray(d['reward']).reshape(1, -1, self.dim_reward))
            next_a.append(np.asarray(d['next_action']).reshape(1, -1, self.dim_action))
            next_s.append(np.asarray(d['next_state']).reshape(1, -1, self.dim_state))
            timestep.append(np.arange(len(d['action'])).reshape(1, -1))
            timestep[-1][timestep[-1] >= self.seq_len] = self.seq_len - 1  # padding cutoff
            if self.config.randomize:
                goal_s.append(np.asarray(d['goal_state']).reshape(1, -1, self.dim_state))

            # padding
            # |FIXME| check padded value & need normalization?
            tlen = o[-1].shape[1]
            o[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_observation)), o[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_action)), a[-1]], axis=1)
            # a[-1] = np.concatenate([np.ones((1, 31 - tlen, 2)) * -100., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_reward)), r[-1]], axis=1)
            next_s[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen, self.dim_state)), next_s[-1]], axis=1)
            timestep[-1] = np.concatenate([np.zeros((1, self.seq_len - tlen)), timestep[-1]], axis=1)
            if self.config.randomize:
                mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen + 1), True, dtype=bool)], axis=1))
            else:
                mask.append(np.concatenate([np.full((1, self.seq_len - tlen), False, dtype=bool), np.full((1, tlen), True, dtype=bool)], axis=1))

        o = th.from_numpy(np.concatenate(o, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        a = th.from_numpy(np.concatenate(a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        r = th.from_numpy(np.concatenate(r, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_a = th.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        next_s = th.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        timestep = th.from_numpy(np.concatenate(timestep, axis=0)).to(dtype=th.long, device=th.device(self.device))
        mask = th.from_numpy(np.concatenate(mask, axis=0)).to(device=th.device(self.device))
        if self.config.randomize:
            goal_s = th.from_numpy(np.concatenate(goal_s, axis=0)).to(dtype=th.float32, device=th.device(self.device))
        
        out = {'observation': o,
            'action': a,
            'reward': r,
            'next_action': next_a,
            'next_state': next_s,
            'timestep': timestep,
            'mask': mask}
        
        if self.config.randomize:
            out['goal_state'] = goal_s

        return out



class LightDarkPreferenceDataset(Dataset):
    '''
    Dataset for preference loss. To compute preference, we must input a pair of data points. The rule of preference is like this:
        1. node in a success trajectory is always better than that of failed trajectory
        2. Among nodes within a success trajectory, one closer to the goal is better
        3. No failed traj vs trailed traj should be compared
    We implement this rule by using two LightDarkDatasets; one for success, one for entire data.
    By sampling one data sample from success trajectory and another from the entire trajectory, we can come up with the preference pair.
    '''
    def __init__(self, success_dataset: Dataset, entire_dataset: Dataset):
        super().__init__()
        self.success_dataset: Dataset = success_dataset
        self.entire_dataset: Dataset = entire_dataset
        self.len_entire_dataset: int = len(entire_dataset)
        self.p_sample_list = success_dataset.p_sample_list
        

    def __len__(self):
        return len(self.success_dataset)

    def __getitem__(self, index):
        
        # get a node from a success trajectory
        success_node = self.success_dataset[index]
        comparison_idx = random.randint(0, self.len_entire_dataset-1)
        comparison_node = self.entire_dataset[comparison_idx]
        
        # check preference to create label
        # label = self._check_preference(success_node, comparison_node)

        data = {'success_node': success_node,
                'comparison_node': comparison_node}

        return data

class PreferenceBatchMaker():
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self._single_batch_maker = BatchMaker(config)

    def __call__(self, data):

        success_nodes, comparison_nodes, preferences = [], [], []
        for datum in data:
            success_nodes.append(datum['success_node'])
            comparison_nodes.append(datum['comparison_node'])

        batch_success_nodes = self._single_batch_maker(success_nodes)
        batch_comparison_nodes = self._single_batch_maker(comparison_nodes)
        preferences = self._check_preference(batch_success_nodes, batch_comparison_nodes)

        out = {'success_node': batch_success_nodes,
               'comparison_node': batch_comparison_nodes,
               'preference': preferences}

        return out

    def _check_preference(self, batch_success_nodes: Dict, batch_comparison_nodes: Dict):
        preferences = th.cat((batch_success_nodes['success_or_fail'], batch_comparison_nodes['success_or_fail']), dim=1)
        
        for i, preference in enumerate(preferences):
            if preference[0] == 1 and preference[1] == 1:
                success_goal_dist = batch_success_nodes['observation'][i].shape[0] - batch_success_nodes['timestep_idx'][i]
                comparison_goal_dist = batch_comparison_nodes['observation'][i].shape[0] - batch_comparison_nodes['timestep_idx'][i]
                if success_goal_dist < comparison_goal_dist:
                    preferences[i][0] = 1
                    preferences[i][1] = 0
                elif success_goal_dist == comparison_goal_dist:
                    preferences[i][0] = 0.5
                    preferences[i][1] = 0.5
                else:
                    preferences[i][0] = 0
                    preferences[i][1] = 1
        
        return preferences



def get_loader(config, dataset: Dict,
               transform=None, collate_fn=None, train=True):
    if config.data_type == 'success':
        dataset = LightDarkDataset(config, dataset, transform)
        if collate_fn == None:
            batcher = BatchMaker(config)
    elif config.data_type == 'mcts':
        dataset = MCTSLightDarkDataset(config, dataset, transform)
        if collate_fn == None:
            batcher = MCTSBatchMaker(config)
    elif config.data_type == 'preference':
        # |TODO|: need to connect config
        # load sub-datasets
        success_files = glob.glob(os.path.join(config.path, f'success{config.pref_data_file}/*')) if train \
            else glob.glob(os.path.join(config.path, f'success{config.pref_data_file}_eval/*'))
        success_dataset = LightDarkDataset(config, success_files, transform=transform)
        entire_files = glob.glob(os.path.join(config.path, f'success{config.pref_data_file}/*')) + glob.glob(os.path.join(config.path, f'fail{config.pref_data_file}/*')) if train \
            else glob.glob(os.path.join(config.path, f'success{config.pref_data_file}_eval/*')) + glob.glob(os.path.join(config.path, f'fail{config.pref_data_file}_eval/*'))
        entire_dataset = LightDarkDataset(config, entire_files, transform=transform)

        dataset = LightDarkPreferenceDataset(success_dataset, entire_dataset)

        if collate_fn == None:
            batcher = PreferenceBatchMaker(config)
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

def get_loader_multi_target(config, dataset):
    dataset = MultiTargetLightDarkDataset(config, dataset)

    loader = DataLoader(dataset, batch_size=1, shuffle=config.shuffle)

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