import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm
import json
from learning.dataset.common import read_trajectory_from_json_and_numpy
from learning.dataset.fetching_value_dataset import get_future_discounted_reward

import pickle


def main():

    DATASET_PATH = "/home/ssd2/sanghyeon/vessl/May6th_3obj_depth8_3000/sim_dataset/train"
    JSON_DATASET_PATH = os.path.join(DATASET_PATH, "dataset_json")
    NPZ_DATASET_PATH = os.path.join(DATASET_PATH, "dataset_numpy")

    annot_filename = "entire.csv"
    annots = pd.read_csv(os.path.join(DATASET_PATH, annot_filename))["filename"].tolist()
    count_fn = Counter(JSON_DATASET_PATH, NPZ_DATASET_PATH)


    record = []

    CPU = 64
    with mp.Pool(processes=CPU) as pool:
        with tqdm(total=len(annots)) as pbar:
            for i, list_stat in enumerate(pool.imap_unordered(count_fn, annots)):
                for j, stat in enumerate(list_stat):
                    
                    if stat is None:
                        continue

                    ( future_discounted_reward_label,
                        is_success,
                        remaining_time_step,
                        is_t_visible ) = stat
        
                    record.append({
                        "future_reward"      : future_discounted_reward_label,
                        "success"            : is_success,
                        "remaining_time_step": remaining_time_step,
                        "is_t_visible"       : is_t_visible})

                pbar.update()
    
    with open("./stat_occ_pickplace.pickle", "wb") as f:
        pickle.dump(record, f)




class Counter:

    def __init__(self, json_path, npz_path):
        self.json_path = json_path
        self.npz_path = npz_path
    

    def __call__(self, fname):
        
        json_fname = os.path.join(self.json_path, f"{fname}.json")
        npz_fname = os.path.join(self.npz_path, f"{fname}.npz")
        with open(json_fname, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_fname)

        # Read trajectory
        init_observation, goal_condition, termination, \
            trajectory_action, trajectory_observation, trajectory_reward,\
            trajectory_length, \
            num_real_exec, num_sim_exec, num_rollout_exec \
                = read_trajectory_from_json_and_numpy(json_data, npz_data)
        
        trajectory_observation_mod = [init_observation, ] + trajectory_observation
        list_stats = []
        if trajectory_length < 3:
            return list_stats

        for time_step in range(2, trajectory_length):

            action_tm1 = trajectory_action[time_step-1]    # PICK
            action_t   = trajectory_action[time_step]      # PLACE
            obs_tm2    = trajectory_observation_mod[time_step-2]   # Occlude
            obs_tm1    = trajectory_observation_mod[time_step]     # Grasp
            obs_t      = trajectory_observation_mod[time_step+1]   # Occlude or not
            
            # Observation at t-2
            _, rgb_tm2, _ = obs_tm2
            is_tm2_visible = check_visibility_from_numpy(rgb_tm2)
            if not is_tm2_visible:
                list_stats.append(None)

            # Action at t-1
            action_tm1_type            = action_tm1[0]
            action_tm1_is_target_bool  = action_tm1[1]
            action_tm1_pos             = action_tm1[2]
            action_tm1_orn_e           = action_tm1[3]
            action_tm1_dyaw            = action_tm1[4]
            #   We are not interested in PLACE or infeasible action or target action
            if action_tm1_type == "PLACE" or action_tm1_pos is None or action_tm1_is_target_bool == True:
                list_stats.append(None)
                continue

            # Observation at t-1
            _, _, grasp_tm1 = obs_tm1
            #   We are not interested in grasp nontarget fail.
            if grasp_tm1 == False:
                list_stats.append(None)
                continue

            # Action at t
            action_t_type            = action_t[0]
            action_t_is_target_bool  = action_t[1]
            action_t_pos             = action_t[2]
            action_t_orn_e           = action_t[3]
            action_t_dyaw            = action_t[4]
            if action_t_type == "PICK" or action_t_pos is None or action_t_is_target_bool == True:
                list_stats.append(None)
                continue

            # Observation at t
            _, rgb_t, _ = obs_t
            is_t_visible = check_visibility_from_numpy(rgb_t)
            
            # Stats...
            #   Future accumulated reward
            trajectory_reward = list(trajectory_reward) + [[0]]   # 0 means the terminal reward
            future_discounted_reward_label \
                = get_future_discounted_reward(future_rewards = trajectory_reward[time_step+1:], 
                                               normalize      = False).item()
            #   Termination success or other
            is_success = True if termination=="success" else False
            #   Remaining time step
            remaining_time_step = trajectory_length - time_step - 1
            
            # Aggregate
            # Aggregate
            stat = (
                future_discounted_reward_label,
                is_success,
                remaining_time_step,
                is_t_visible)
            list_stats.append(stat)

        return list_stats


def check_visibility_from_numpy(observation_rgb_image):
    """Assume the data is already foreground segmented"""
    is_target_visible = bool(np.sum((observation_rgb_image[:,:,0]>1)))

    return is_target_visible




if __name__=="__main__":
    main()



        

