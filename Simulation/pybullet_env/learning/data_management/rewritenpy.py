from read_hierarchical_annot import read_hierarchical_annotations
import pickle
import os
import json
import numpy as np

def rewrite(fname):


    fpath = os.path.join(DATA_PATH, fname)

    with open(fpath, "rb") as f:
        data = pickle.load(f)

        # Converting type and dropping unnecesary infos.
        #   init_observation
        depth_img = np.array(data["init_observation"][0])
        rgb_img = np.array(data["init_observation"][1])
        seg_mask = data["init_observation"][2]
        seg_target = seg_mask["O"]
        seg_other = seg_mask["X"]
        seg_all = np.logical_or(seg_target, seg_other)
        masked_depth_img = np.copy(depth_img)
        masked_depth_img[~seg_all] = 0.0
        masked_rgb_img = np.copy(rgb_img)
        masked_rgb_img[~seg_all] = 0.0
        data["init_observation"] \
            = (masked_depth_img, masked_rgb_img)
        #   exec_action
        exec_action_list = []
        for exec_action in data["exec_action"]:
            exec_action = exec_action[:5] # Discard traj
            exec_action_list.append(exec_action)
        data["exec_action"] = exec_action_list
        #   exec_observation
        exec_obs_list = []
        for exec_obs in data["exec_observation"]:
            exec_obs_entry = (np.array(exec_obs[0]), np.array(exec_obs[1]))
            exec_obs_list.append(exec_obs_entry)
        data["exec_observation"] = exec_obs_list
        #   exec_reward
        
        #   sim_action
        sim_action_list = []
        for sim_action in data["sim_action"]:
            sim_action = sim_action[:5] # Discard traj
            sim_action_list.append(sim_action)
        data["sim_action"] = sim_action_list
        #   sim_obsrvation
        sim_obs_list = []
        for sim_obs in data["sim_observation"]:
            sim_obs_entry = (np.array(sim_obs[0]), np.array(sim_obs[1]))
            sim_obs_list.append(sim_obs_entry)
        data["sim_observation"] = sim_obs_list
        #   sim_reward




        new_data = dict()
        new_data["goal_condition"] = data["goal_condition"]
        new_data["exec_action"] = data["exec_action"]
        new_data["exec_reward"] = data["exec_reward"]
        new_data["sim_action"] = data["sim_action"]
        new_data["sim_reward"] = data["sim_reward"]

        data_json = json.dumps(new_data)
        num_real_exec = len(exec_action_list)
        num_sim_exec = len(sim_action_list)
        data_npz_kwargs = {}
        data_npz_kwargs["init_observation_depth"] = data["init_observation"][0]
        data_npz_kwargs["init_observation_rgb"] = data["init_observation"][1]
        for i in range(num_real_exec):
            data_npz_kwargs[f"exec_observation_{i}_depth"] = data["exec_observation"][i][0]
            data_npz_kwargs[f"exec_observation_{i}_rgb"] = data["exec_observation"][i][1]
        for i in range(num_sim_exec):
            data_npz_kwargs[f"sim_observation_{i}_depth"] = data["sim_observation"][i][0]
            data_npz_kwargs[f"sim_observation_{i}_rgb"] = data["sim_observation"][i][1]

        fname_cut = os.path.splitext(fname.split('/')[1])[0]
        json_file_path = os.path.join(SAVE_PATH_JSON, fname_cut+".json")
        with open(json_file_path, 'w') as f:
            json.dump(data_json, f)
        npz_file_path = os.path.join(SAVE_PATH_NPZ, fname_cut+".npz")
        np.savez(npz_file_path, **data_npz_kwargs)
        


if __name__=="__main__":
    # DATA_PATH = "/home/sanghyeon/vessl/cql/sim_dataset_hierarchy"
    DATA_PATH = "/home/sanghyeon/vessl/cql/sim_dataset_fail_hierarchy"
    SAVE_PATH_NPZ = "/home/sanghyeon/vessl/cql/sim_negative_npz"
    SAVE_PATH_JSON = "/home/sanghyeon/vessl/cql/sim_negative_json"


    # Train/eval split
    annots = read_hierarchical_annotations(DATA_PATH)

    import multiprocessing as mp
    from tqdm import tqdm
    with mp.Pool(processes=256) as pool:
        with tqdm(total=len(annots)) as pbar:
            for i in enumerate(pool.imap_unordered(rewrite, annots)):
                pbar.update()