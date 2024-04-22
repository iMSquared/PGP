import os
import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import List, Dict, Tuple, Union

from learning.model.value import HistoryPlaceValueonly
from learning.dataset.common import reconstruct_traj_from_json_and_numpy
                                    


def main():

    FILENAMES       = pd.read_csv("/home/jiyong/vessl/April30th1000_cuboid12_horizon4/sim_dataset/train/entire.csv")["filename"].tolist()
    DATA_PATH_JSON  = "/home/jiyong/vessl/April30th1000_cuboid12_horizon4/sim_dataset/train/dataset_json"
    DATA_PATH_NUMPY = "/home/jiyong/vessl/April30th1000_cuboid12_horizon4/sim_dataset/train/dataset_numpy"
    NUM_PROCESS     = 64

    total_place = 0
    total_place_success = 0
    total_invisible_count = 0
    total_invisible_success_count = 0
    imap_fn = ScanReoccludingPlace(DATA_PATH_JSON, DATA_PATH_NUMPY)

    with mp.Pool(processes=NUM_PROCESS) as pool:
        with tqdm(total=len(FILENAMES)) as pbar:
            for i, (num_place, num_place_success_count, num_invisible_count, num_inivisible_success_count) in enumerate(pool.imap_unordered(imap_fn, FILENAMES)):
                # Count
                total_place += num_place
                total_place_success += num_place_success_count
                total_invisible_count += num_invisible_count
                total_invisible_success_count += num_inivisible_success_count
                # Log
                pbar.update()
                # pbar.set_postfix(total=total_place, count=total_invisible_count # Slow...
                
    print(f"total: {total_place}, total_success: {total_place_success}, invisible: {total_invisible_count}, invisible_success: {total_invisible_success_count}")






class ScanReoccludingPlace:

    def __init__(self, data_path_json, data_path_numpy):
        self.data_path_json = data_path_json
        self.data_path_numpy = data_path_numpy


    def __call__(self, fname) -> Tuple[int, int]:
        # Open
        json_file_name = os.path.join(self.data_path_json, f"{fname}.json")
        npz_file_name = os.path.join(self.data_path_numpy, f"{fname}.npz")
        with open(json_file_name, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_file_name)
        # Read traj
        init_observation, goal_condition, termination, \
            trajectory_action, trajectory_observation, trajectory_reward, \
            trajectory_length = reconstruct_traj_from_json_and_numpy(json_data, npz_data)
        # Find PLACE(non-target) action
        num_place = 0
        num_place_success_count = 0
        num_invisible_count = 0
        num_invisible_success_count = 0
        for i, (action, observation, reward) in \
                enumerate(zip(trajectory_action, trajectory_observation, trajectory_reward)):
            # Parse action
            action_type           = action[0]
            action_is_target_bool = action[1]
            action_pos            = action[2]
            action_orn_e          = action[3]
            action_dyaw           = action[4]
            # Select PLACE(non-target)
            if action_type == "PLACE" and action_is_target_bool == False:
                num_place += 1
                if termination == "success":
                    num_place_success_count += 1
                num_place += 1
                # Check observation
                observation_depth_image, observation_rgb_image, observation_grasp = observation
                target_visibility = check_fullocclusion(observation_rgb_image)

                # Count
                if not target_visibility:
                    num_invisible_count += 1
                    # plt.imshow(observation_rgb_image)
                    # plt.show()
                    if termination == "success":
                        num_invisible_success_count += 1

        return num_place, num_place_success_count, num_invisible_count, num_invisible_success_count



def check_fullocclusion(observation_rgb_image):
    is_target_visible = bool(np.sum((observation_rgb_image[:,:,0]>1)))

    return is_target_visible







if __name__=="__main__":
     main()