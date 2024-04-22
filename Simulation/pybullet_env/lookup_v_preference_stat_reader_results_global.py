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

    with open("./exp_fetching/stat_at_2.pickle", "rb") as f:
        list_stat = pickle.load(f)

    occluded_pick_target_stat    = []
    occluded_pick_nontarget_stat = []
    visible_pick_target_stat     = []
    visible_pick_nontarget_stat  = []
    # Aggregate all time step... preference do not really care about the time.
    for item in list_stat:
        occluded_pick_target_stat    += item["occluded"]["pick_target"]
        occluded_pick_nontarget_stat += item["occluded"]["pick_nontarget"]
        visible_pick_target_stat     += item["visible"]["pick_target"]
        visible_pick_nontarget_stat  += item["visible"]["pick_nontarget"]

    occluded_pick_target_df = pd.DataFrame(occluded_pick_target_stat)
    occluded_pick_nontarget_df = pd.DataFrame(occluded_pick_nontarget_stat)
    visible_pick_target_df = pd.DataFrame(visible_pick_target_stat)
    visible_pick_nontarget_df = pd.DataFrame(visible_pick_nontarget_stat)

    # Augment success labels.
    results_gr_occluded_pick_target_stat    = occluded_pick_target_df.groupby("success")
    results_gr_occluded_pick_nontarget_stat = occluded_pick_nontarget_df.groupby("success")
    results_gr_visible_pick_target_stat     = visible_pick_target_df.groupby("success")
    results_gr_visible_pick_nontarget_stat  = visible_pick_nontarget_df.groupby("success")
    success_gr_occluded_pick_target_df = results_gr_occluded_pick_target_stat.get_group(True)
    success_gr_occluded_pick_nontarget_stat = results_gr_occluded_pick_nontarget_stat.get_group(True)
    success_gr_visible_pick_target_stat = results_gr_visible_pick_target_stat.get_group(True)
    success_gr_visible_pick_nontarget_stat = results_gr_visible_pick_nontarget_stat.get_group(True)

    occluded_pick_target_df    = pd.concat([occluded_pick_target_df, success_gr_occluded_pick_target_df, success_gr_occluded_pick_target_df, success_gr_occluded_pick_target_df, success_gr_occluded_pick_target_df])
    occluded_pick_nontarget_df = pd.concat([occluded_pick_nontarget_df, success_gr_occluded_pick_nontarget_stat, success_gr_occluded_pick_nontarget_stat, success_gr_occluded_pick_nontarget_stat, success_gr_occluded_pick_nontarget_stat])
    visible_pick_target_df     = pd.concat([visible_pick_target_df, success_gr_visible_pick_target_stat, success_gr_visible_pick_target_stat, success_gr_visible_pick_target_stat, success_gr_visible_pick_target_stat])
    visible_pick_nontarget_df  = pd.concat([visible_pick_nontarget_df, success_gr_visible_pick_nontarget_stat, success_gr_visible_pick_nontarget_stat, success_gr_visible_pick_nontarget_stat, success_gr_visible_pick_nontarget_stat, success_gr_visible_pick_nontarget_stat])



    # Occluded pick target success
    occ_pick_target_groupedby_graspsuccess = occluded_pick_target_df.groupby("grasp")
    occ_pick_target_graspsuccess_gr = occ_pick_target_groupedby_graspsuccess.get_group(True)
    occ_pick_target_graspfail_gr = occ_pick_target_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Label distribution at time=2")
    
    num_success = 0
    num_fail = 0
    success_timesteps = []
    fail_timesteps = []

    success_avg = np.mean(occ_pick_target_graspsuccess_gr["success"]).item()
    total_data = len(occ_pick_target_graspsuccess_gr["success"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Occlusion - PICK(target) - grasp success\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_target_graspsuccess_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    print(f"num success: {num_success}")
    print(f"num fail: {num_fail}")
    print(f"num total: {num_success + num_fail}")
    print(f"average remaining timesteps in success: {np.mean(success_timesteps)}")
    print(f"average remaining timesteps in fail: {np.mean(fail_timesteps)}")
    print(f"average remaining timesteps total: {np.mean(success_timesteps+fail_timesteps)}")


    num_success = 0
    num_fail = 0
    success_timesteps = []
    fail_timesteps = []

    success_avg = np.mean(occ_pick_target_graspfail_gr["success"]).item()
    total_data = len(occ_pick_target_graspfail_gr["success"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Occlusion - PICK(target) - grasp fail\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_target_graspfail_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    plt.show()

    print(f"num success: {num_success}")
    print(f"num fail: {num_fail}")
    print(f"num total: {num_success + num_fail}")
    print(f"average remaining timesteps in success: {np.mean(success_timesteps)}")
    print(f"average remaining timesteps in fail: {np.mean(fail_timesteps)}")
    print(f"average remaining timesteps total: {np.mean(success_timesteps+fail_timesteps)}")



    # Occluded pick nontarget success
    occ_pick_nontarget_groupedby_graspsuccess = occluded_pick_nontarget_df.groupby("grasp")
    occ_pick_nontarget_graspsuccess_gr = occ_pick_nontarget_groupedby_graspsuccess.get_group(True)
    occ_pick_nontarget_graspfail_gr = occ_pick_nontarget_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Label distribution at time=2")
    


    num_success = 0
    num_fail = 0
    success_timesteps = []
    fail_timesteps = []

    success_avg = np.mean(occ_pick_nontarget_graspsuccess_gr["success"]).item()
    total_data = len(occ_pick_nontarget_graspsuccess_gr["success"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Occlusion - PICK(nontarget) - grasp success\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_nontarget_graspsuccess_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)


    print(f"num success: {num_success}")
    print(f"num fail: {num_fail}")
    print(f"num total: {num_success + num_fail}")
    print(f"average remaining timesteps in success: {np.mean(success_timesteps)}")
    print(f"average remaining timesteps in fail: {np.mean(fail_timesteps)}")
    print(f"average remaining timesteps total: {np.mean(success_timesteps+fail_timesteps)}")



    success_avg = np.mean(occ_pick_nontarget_graspfail_gr["success"]).item()
    total_data = len(occ_pick_nontarget_graspfail_gr["success"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Occlusion - PICK(nontarget) - grasp fail\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = occ_pick_nontarget_graspfail_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    plt.show()



    # Visible pick target success
    vis_pick_target_groupedby_graspsuccess = visible_pick_target_df.groupby("grasp")
    vis_pick_target_graspsuccess_gr = vis_pick_target_groupedby_graspsuccess.get_group(True)
    vis_pick_target_graspfail_gr = vis_pick_target_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Label distribution at time=2")
    
    success_avg = np.mean(vis_pick_target_graspsuccess_gr["success"]).item()
    total_data = len(vis_pick_target_graspsuccess_gr["success"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Visible - PICK(target) - grasp success\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = vis_pick_target_graspsuccess_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    success_avg = np.mean(vis_pick_target_graspfail_gr["success"]).item()
    total_data = len(vis_pick_target_graspfail_gr["success"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Visible - PICK(target) - grasp fail\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = vis_pick_target_graspfail_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    plt.show()


    # Visible pick target success
    vis_pick_nontarget_groupedby_graspsuccess = visible_pick_nontarget_df.groupby("grasp")
    vis_pick_nontarget_graspsuccess_gr = vis_pick_nontarget_groupedby_graspsuccess.get_group(True)
    vis_pick_nontarget_graspfail_gr = vis_pick_nontarget_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Label distribution at time=2")
    
    success_avg = np.mean(vis_pick_nontarget_graspsuccess_gr["success"]).item()
    total_data = len(vis_pick_nontarget_graspsuccess_gr["success"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Visible - PICK(nontarget) - grasp success\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = vis_pick_nontarget_graspsuccess_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    success_avg = np.mean(vis_pick_nontarget_graspfail_gr["success"]).item()
    total_data = len(vis_pick_nontarget_graspfail_gr["success"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Visible - PICK(nontarget) - grasp fail\nAvg={success_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    
    groupby_t = vis_pick_nontarget_graspfail_gr.groupby("remaining_time_step")
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
    ax.set_zlim(0, 800000)

    plt.show()








if __name__=="__main__":
    main()



        

