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


    list_stat_record = [ {
        "occluded": {
            "pick_target": [],
            "pick_nontarget": []},
        "visible": {
            "pick_target": [],
            "pick_nontarget": [] } } for i in range(8)]


    CPU = 64
    with mp.Pool(processes=CPU) as pool:
        with tqdm(total=len(annots)) as pbar:
            for i, list_stat in enumerate(pool.imap_unordered(count_fn, annots)):
                for j, stat in enumerate(list_stat):
                    
                    if stat is None:
                        continue

                    ( is_visible,
                        is_pick_target,
                        grasp_at_t,
                        future_discounted_reward_label,
                        is_success,
                        remaining_time_step,
                        remaining_picks  ) = stat

                
                    if not is_visible and is_pick_target:
                        list_stat_record[j]["occluded"]["pick_target"].append({
                            "future_reward"      : future_discounted_reward_label,
                            "success"            : is_success,
                            "grasp"              : grasp_at_t,
                            "remaining_time_step": remaining_time_step,
                            "remaining_picks"    : remaining_picks })
                    elif not is_visible and not is_pick_target:
                        list_stat_record[j]["occluded"]["pick_nontarget"].append({
                            "future_reward"      : future_discounted_reward_label,
                            "success"            : is_success,
                            "grasp"              : grasp_at_t,
                            "remaining_time_step": remaining_time_step,
                            "remaining_picks"    : remaining_picks })
                    elif is_visible and is_pick_target:
                        list_stat_record[j]["visible"]["pick_target"].append({
                            "future_reward"      : future_discounted_reward_label,
                            "success"            : is_success,
                            "grasp"              : grasp_at_t,
                            "remaining_time_step": remaining_time_step,
                            "remaining_picks"    : remaining_picks })
                    else:
                        list_stat_record[j]["visible"]["pick_nontarget"].append({
                            "future_reward"      : future_discounted_reward_label,
                            "success"            : is_success,
                            "grasp"              : grasp_at_t,
                            "remaining_time_step": remaining_time_step,
                            "remaining_picks"    : remaining_picks })
                
                pbar.update()
    
    with open("./stat_at_2.pickle", "wb") as f:
        pickle.dump(list_stat_record, f)




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
        for time_step in range(0, trajectory_length):
            action_at_t     = trajectory_action[time_step]
            obs_at_t_minus1 = trajectory_observation_mod[time_step]
            obs_at_t        = trajectory_observation_mod[time_step+1]
            # observation at t-1
            depth_at_t_minus1, rgb_at_t_minus1, grasp_at_t_minus1 = obs_at_t_minus1
            is_visible = check_visibility_from_numpy(rgb_at_t_minus1)
            # action at t
            action_type            = action_at_t[0]
            action_is_target_bool  = action_at_t[1]
            action_pos             = action_at_t[2]
            action_orn_e           = action_at_t[3]
            action_dyaw            = action_at_t[4]
            if action_type == "PICK" and action_pos is not None:
                if action_is_target_bool:
                    is_pick_target = True
                else:
                    is_pick_target = False
            else:
                list_stats.append(None)   # Infeasible action or place...
                continue
            # observation at t
            depth_at_t, rgb_at_t, grasp_at_t = obs_at_t

            # Stats...
            # future accumulated reward
            trajectory_reward = list(trajectory_reward) + [[0]]   # 0 means the terminal reward
            future_discounted_reward_label \
                = get_future_discounted_reward(future_rewards = trajectory_reward[time_step+1:], 
                                               normalize      = False).item()
            # success or other
            is_success = True if termination=="success" else False
            # remaining time step
            remaining_time_step = trajectory_length - time_step - 1
            # remaining picks
            if time_step+1 >= trajectory_length:
                remaining_picks = 0
            else:
                remaining_picks = 0
                for future_action in trajectory_action[time_step+1:]:
                    action_type = future_action[0]
                    if action_type == "PICK":
                        remaining_picks += 1

            # Aggregate
            stat = (
                is_visible,
                is_pick_target,
                grasp_at_t.item(),
                future_discounted_reward_label,
                is_success,
                remaining_time_step,
                remaining_picks )
            list_stats.append(stat)

        return list_stats


def check_visibility_from_numpy(observation_rgb_image):
    """Assume the data is already foreground segmented"""
    is_target_visible = bool(np.sum((observation_rgb_image[:,:,0]>1)))

    return is_target_visible




if __name__=="__main__":
    main()



        

