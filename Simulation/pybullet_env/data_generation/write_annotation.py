import os
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def main():
    save_path_root = "/home/shared_directory/vessl/May6th_3obj_depth8_3000/sim_dataset/eval"
    json_dir_path = os.path.join(save_path_root, "dataset_json")
    numpy_dir_path = os.path.join(save_path_root, "dataset_numpy")

    list_entiredata_name = []
    list_entiredata_num_subdata = []
    list_successdata_name = []
    list_successdata_num_subdata = []
    list_faildata_name = []
    list_faildata_num_subdata = []
    list_policydata_name = []
    list_policydata_num_subdata = []

    
    json_filenames = sorted(os.listdir(json_dir_path))
    for j_fname in tqdm(json_filenames):
        
        with open(os.path.join(json_dir_path, j_fname), "r") as f:
            data = json.load(f)

        trajectory_action = data["exec_action"] + data["sim_action"] + data["rollout_action"]
        num_actions = len(trajectory_action)
        num_subdata = num_actions + 1 # Root of the planning tree to the sim node.

        fname = Path(j_fname).stem    # Without extension
        list_entiredata_name.append(fname)
        list_entiredata_num_subdata.append(num_subdata)
        if data["termination"] == "success":
            list_successdata_name.append(fname)
            list_successdata_num_subdata.append(num_subdata)
        else:
            list_faildata_name.append(fname)
            list_faildata_num_subdata.append(num_subdata)

        if validate_place_existance(trajectory_action):
            list_policydata_name.append(fname)
            list_policydata_num_subdata.append(num_subdata)

        

    entireset_df = pd.DataFrame(zip(list_entiredata_name, list_entiredata_num_subdata),\
                            columns=["filename", "num_subdata"])
    entireset_df.to_csv(path_or_buf=os.path.join(save_path_root, "entire.csv"))
    
    successset_df = pd.DataFrame(zip(list_successdata_name, list_successdata_num_subdata),\
                             columns=["filename", "num_subdata"])
    successset_df.to_csv(path_or_buf=os.path.join(save_path_root, "success.csv"))

    failset_df = pd.DataFrame(zip(list_faildata_name, list_faildata_num_subdata),\
                             columns=["filename", "num_subdata"])
    failset_df.to_csv(path_or_buf=os.path.join(save_path_root, "fail.csv"))

    policyset_df = pd.DataFrame(zip(list_policydata_name, list_policydata_num_subdata),\
                             columns=["filename", "num_subdata"])
    policyset_df.to_csv(path_or_buf=os.path.join(save_path_root, "policy.csv"))



def validate_place_existance(trajectory_action) -> bool:
    """Validate existance of PLACE action in the trajectory
    
    Args:
        trajectory_action(List): Action trajectory
    
    Returns:
        bool: True if exist
    """
    is_place_exist = False
    for action in trajectory_action:
        action_type = action[0]
        action_pos = action[2]
        if action_type == "PLACE" and action_pos is not None:
            is_place_exist = True
    
    return is_place_exist





if __name__=="__main__":
    main()