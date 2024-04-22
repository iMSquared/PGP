import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm
import json
from learning.dataset.common import read_trajectory_from_json_and_numpy
from learning.dataset.fetching_value_dataset import get_future_discounted_reward
import matplotlib.pyplot as plt


import pickle


def main():

    with open("./exp_fetching/stat_occ_pickplace.pickle", "rb") as f:
        list_stat = pickle.load(f)
    occluded_pick_nontarget_df = pd.DataFrame(list_stat)
    print(occluded_pick_nontarget_df)

    # Occluded pick target success
    occ_pick_target_nontarget_visiblity = occluded_pick_nontarget_df.groupby("is_t_visible")
    occ_pick_target_nontarget_visible_gr = occ_pick_target_nontarget_visiblity.get_group(True)
    occ_pick_target_nontarget_occluded_gr = occ_pick_target_nontarget_visiblity.get_group(False)



    # Visible group
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Label distribution")
    
    visible_avg = np.mean(occ_pick_target_nontarget_visible_gr["success"]).item()
    total_data = len(occ_pick_target_nontarget_visible_gr["success"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Occlusion - PICK(non-target) - PLACE - visible\nAvg={visible_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_target_nontarget_visible_gr.groupby("remaining_time_step")
    for remaining_t in range(0, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_result = t_group["success"]
        try:
            t_success = t_result.value_counts()[True]
        except:
            t_success = 0
        try:
            t_fail = t_result.value_counts()[False]
        except:
            t_fail = 0
        ax.bar(["Fail", "Success"] , [t_fail, t_success], zs=remaining_t, zdir='y', alpha=0.8, width=0.2)
    ax.set_xlabel('Result')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_zlim(0, 400000)

    num_success = 0
    num_fail = 0
    success_timesteps = []
    fail_timesteps = []


    occluded_avg = np.mean(occ_pick_target_nontarget_occluded_gr["success"]).item()
    total_data = len(occ_pick_target_nontarget_occluded_gr["success"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Occlusion - PICK(non-target) - PLACE - invisible\nAvg={occluded_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_target_nontarget_occluded_gr.groupby("remaining_time_step")
    for remaining_t in range(0, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_result = t_group["success"]
        try:
            t_success = t_result.value_counts()[True]
            num_success += t_success
            for i in range(t_success):
                success_timesteps.append(remaining_t)        
        except:
            t_success = 0
        try:
            t_fail = t_result.value_counts()[False]
            num_fail += t_fail
            for i in range(t_fail):
                fail_timesteps.append(remaining_t)        
        except:
            t_fail = 0
        ax.bar(["Fail", "Success"] , [t_fail, t_success], zs=remaining_t, zdir='y', alpha=0.8, width=0.2)
    ax.set_xlabel('Result')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.set_zlim(0, 400000)

    plt.show()







if __name__=="__main__":
    main()



        

