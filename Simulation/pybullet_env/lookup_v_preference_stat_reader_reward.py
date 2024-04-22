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

    with open("./stat_at_2.pickle", "rb") as f:
        list_stat = pickle.load(f)


    occluded_pick_target_stat    = list_stat[2]["occluded"]["pick_target"]
    occluded_pick_nontarget_stat = list_stat[2]["occluded"]["pick_nontarget"]
    visible_pick_target_stat     = list_stat[2]["visible"]["pick_target"]
    visible_pick_nontarget_stat  = list_stat[2]["visible"]["pick_nontarget"]

    occluded_pick_target_df = pd.DataFrame(occluded_pick_target_stat)
    occluded_pick_nontarget_df = pd.DataFrame(occluded_pick_nontarget_stat)
    visible_pick_target_df = pd.DataFrame(visible_pick_target_stat)
    visible_pick_nontarget_df = pd.DataFrame(visible_pick_nontarget_stat)


    # Occluded pick target value
    occ_pick_target_groupedby_graspsuccess = occluded_pick_target_df.groupby("grasp")
    occ_pick_target_graspsuccess_gr = occ_pick_target_groupedby_graspsuccess.get_group(True)
    occ_pick_target_graspfail_gr = occ_pick_target_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Future reward distribution at time=2")
    
    reward_avg = np.mean(occ_pick_target_graspsuccess_gr["future_reward"]).item()
    total_data = len(occ_pick_target_graspsuccess_gr["future_reward"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Occlusion - PICK(target) - grasp success\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10

    groupby_t = occ_pick_target_graspsuccess_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)

    reward_avg = np.mean(occ_pick_target_graspfail_gr["future_reward"]).item()
    total_data = len(occ_pick_target_graspfail_gr["future_reward"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Occlusion - PICK(target) - grasp fail\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10
    
    groupby_t = occ_pick_target_graspfail_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)

    plt.show()


    # Occluded pick nontarget value
    occ_pick_nontarget_groupedby_graspsuccess = occluded_pick_nontarget_df.groupby("grasp")
    occ_pick_nontarget_graspsuccess_gr = occ_pick_nontarget_groupedby_graspsuccess.get_group(True)
    occ_pick_nontarget_graspfail_gr = occ_pick_nontarget_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Future reward distribution at time=2")

    reward_avg = np.mean(occ_pick_nontarget_graspsuccess_gr["future_reward"]).item()
    total_data = len(occ_pick_nontarget_graspsuccess_gr["future_reward"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Occlusion - PICK(nontarget) - grasp success\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10

    groupby_t = occ_pick_nontarget_graspsuccess_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)


    reward_avg = np.mean(occ_pick_nontarget_graspfail_gr["future_reward"]).item()
    total_data = len(occ_pick_nontarget_graspfail_gr["future_reward"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Occlusion - PICK(nontarget) - grasp fail\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10
    
    groupby_t = occ_pick_nontarget_graspfail_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)

    plt.show()



    # Visible pick target value
    vis_pick_target_groupedby_graspsuccess = visible_pick_target_df.groupby("grasp")
    vis_pick_target_graspsuccess_gr = vis_pick_target_groupedby_graspsuccess.get_group(True)
    vis_pick_target_graspfail_gr = vis_pick_target_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Future reward distribution at time=2")

    reward_avg = np.mean(vis_pick_target_graspsuccess_gr["future_reward"]).item()
    total_data = len(vis_pick_target_graspsuccess_gr["future_reward"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Visible - PICK(target) - grasp success\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10

    groupby_t = vis_pick_target_graspsuccess_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)


    reward_avg = np.mean(vis_pick_target_graspfail_gr["future_reward"]).item()
    total_data = len(vis_pick_target_graspfail_gr["future_reward"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Visible - PICK(target) - grasp fail\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10
    
    groupby_t = vis_pick_target_graspfail_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)

    plt.show()


    # Visible pick nontarget value
    vis_pick_nontarget_groupedby_graspsuccess = visible_pick_nontarget_df.groupby("grasp")
    vis_pick_nontarget_graspsuccess_gr = vis_pick_nontarget_groupedby_graspsuccess.get_group(True)
    vis_pick_nontarget_graspfail_gr = vis_pick_nontarget_groupedby_graspsuccess.get_group(False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Future reward distribution at time=2")

    reward_avg = np.mean(vis_pick_nontarget_graspsuccess_gr["future_reward"]).item()
    total_data = len(vis_pick_nontarget_graspsuccess_gr["future_reward"])
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(f"Visible - PICK(nontarget) - grasp success\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10

    groupby_t = vis_pick_nontarget_graspsuccess_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)


    reward_avg = np.mean(vis_pick_nontarget_graspfail_gr["future_reward"]).item()
    total_data = len(vis_pick_nontarget_graspfail_gr["future_reward"])
    ax = fig.add_subplot(122, projection='3d')
    ax.set_title(f"Visible - PICK(nontarget) - grasp fail\nAvg={reward_avg:.2f}, # total data={total_data}")
    ax.view_init(elev=35, azim=-135, roll=0)
    nbins=10
    
    groupby_t = vis_pick_nontarget_graspfail_gr.groupby("remaining_time_step")
    for remaining_t in range(1, 8):
        if remaining_t not in groupby_t.groups.keys():
            continue
        t_group = groupby_t.get_group(remaining_t)
        t_rewards = t_group["future_reward"]
        hist, bins = np.histogram(t_rewards, bins=nbins, range=(-175, 25))
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=remaining_t, zdir='y', alpha=0.8, width=bins[1]-bins[0])
    ax.set_xlabel('Reward')
    ax.set_ylabel('Remaining time step')
    ax.set_zlabel('# data')
    ax.set_xlim(-175, 25)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 100000)

    plt.show()


if __name__=="__main__":
    main()



        
