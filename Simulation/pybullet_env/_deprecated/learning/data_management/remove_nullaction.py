import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

import multiprocessing as mp

# Dataset
dataname: str = "April17th"
sim_or_exec: str = "sim_dataset"
file_path_train_entire_annotation: str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/entire.csv"
data_path_train_dataset_json     : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
data_path_train_dataset_npz      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
file_path_eval_entire_annotation : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/entire.csv"
data_path_eval_dataset_json      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
data_path_eval_dataset_npz       : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"


# Train/eval split
train_entire_annots      = pd.read_csv(file_path_train_entire_annotation)["filename"].tolist()
train_entire_num_subdata = pd.read_csv(file_path_train_entire_annotation)["num_subdata"].tolist()
eval_entire_annots       = pd.read_csv(file_path_eval_entire_annotation)["filename"].tolist()
eval_entire_num_subdata  = pd.read_csv(file_path_eval_entire_annotation)["num_subdata"].tolist()




def main():

    cpu = 16
    with mp.Pool(processes=cpu) as pool:
        with tqdm(total=len(train_entire_annots)) as pbar:
            
            count = 0

            for has_null in pool.imap_unordered(check, train_entire_annots):
                if has_null==True:
                    count += 1
                pbar.update()
        
    print(count)


def check(name):
    # Loading json and numpy
    json_file_name = os.path.join(data_path_train_dataset_json, f"{name}.json")
    npz_file_name = os.path.join(data_path_train_dataset_npz, f"{name}.npz")
    with open(json_file_name, "r") as f:
        json_data = json.load(f)
    # npz_data = np.load(npz_file_name)

    # asdf
    has_null = False
    trajectory_action = json_data["exec_action"]+json_data["sim_action"]+json_data["rollout_action"]
    for a in trajectory_action:
        if a[1] == None:
            has_null = True
            break

    if has_null:
        print(name)
        return True
    
    return False


if __name__=="__main__":
    main()