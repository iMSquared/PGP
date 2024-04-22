import os
import json
import numpy as np
import numpy.typing as npt
import torch

from typing import List, Dict, Tuple, Union

import shutil

from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import random

import pandas as pd
from tqdm import tqdm


dataname: str = "May2nd_cuboid12_large"
sim_or_exec: str = "sim_dataset"
    
file_path_annotation: str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/entire.csv"
data_path_json     : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/dataset_json"
data_path_npz      : str = f"/home/jiyong/vessl/{dataname}/{sim_or_exec}/train/dataset_numpy"
data_annots      = pd.read_csv(file_path_annotation)["filename"].tolist()

ill_data = []
# Loading json and numpy
for fname in tqdm(data_annots):
    
    json_file_name = os.path.join(data_path_json, f"{fname}.json")
    npz_file_name = os.path.join(data_path_npz, f"{fname}.npz")
    with open(json_file_name, "r") as f:
        json_data = json.load(f)
    try:
        npz_data = np.load(npz_file_name)
    except:
        ill_data.append(fname)
print(ill_data)