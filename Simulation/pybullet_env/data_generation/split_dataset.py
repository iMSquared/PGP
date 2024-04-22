import os
from pathlib import Path
from tqdm import tqdm

# Path('/root/dir/sub/file.ext').stem
import shutil


src_dir = "/home/jiyong/vessl/May2nd_cuboid12_large"
exec_json_src_dir  = os.path.join(src_dir, "exec_dataset_json")
exec_numpy_src_dir = os.path.join(src_dir, "exec_dataset_numpy")
sim_json_src_dir   = os.path.join(src_dir, "sim_dataset_json")
sim_numpy_src_dir  = os.path.join(src_dir, "sim_dataset_numpy")

exec_dst_dir   = os.path.join(src_dir, "exec_dataset")
sim_dst_dir    = os.path.join(src_dir, "sim_dataset")

ratio = 0.1

if not os.path.exists(exec_dst_dir):
    os.mkdir(exec_dst_dir)
    os.mkdir(os.path.join(exec_dst_dir, "train"))
    os.mkdir(os.path.join(exec_dst_dir, "train", "dataset_json"))
    os.mkdir(os.path.join(exec_dst_dir, "train", "dataset_numpy"))
    os.mkdir(os.path.join(exec_dst_dir, "eval"))
    os.mkdir(os.path.join(exec_dst_dir, "eval", "dataset_json"))
    os.mkdir(os.path.join(exec_dst_dir, "eval", "dataset_numpy"))

if not os.path.exists(sim_dst_dir):
    os.mkdir(sim_dst_dir)
    os.mkdir(os.path.join(sim_dst_dir, "train"))
    os.mkdir(os.path.join(sim_dst_dir, "train", "dataset_json"))
    os.mkdir(os.path.join(sim_dst_dir, "train", "dataset_numpy"))
    os.mkdir(os.path.join(sim_dst_dir, "eval"))
    os.mkdir(os.path.join(sim_dst_dir, "eval", "dataset_json"))
    os.mkdir(os.path.join(sim_dst_dir, "eval", "dataset_numpy"))
    

# Splitting execution set
exec_annots = [Path(v).stem for v in sorted(os.listdir(os.path.join(exec_json_src_dir)))]
num_exec_eval_annots = int(len(exec_annots) * ratio)
print(f"# of eval data in exec: {num_exec_eval_annots}")

# Move trainset
for annot in exec_annots[:-num_exec_eval_annots]:
    # Move json
    data_path = os.path.join(exec_json_src_dir, f"{annot}.json")
    shutil.move(data_path, os.path.join(exec_dst_dir, "train", "dataset_json"))
    # Move numpy
    data_path = os.path.join(exec_numpy_src_dir, f"{annot}.npz")
    shutil.move(data_path, os.path.join(exec_dst_dir, "train", "dataset_numpy"))

# Move evalset
for annot in exec_annots[-num_exec_eval_annots:]:
    # Move json
    data_path = os.path.join(exec_json_src_dir, f"{annot}.json")
    shutil.move(data_path, os.path.join(exec_dst_dir, "eval", "dataset_json"))
    # Move numpy
    data_path = os.path.join(exec_numpy_src_dir, f"{annot}.npz")
    shutil.move(data_path, os.path.join(exec_dst_dir, "eval", "dataset_numpy"))


# Splitting simulation set
sim_annots = [Path(v).stem for v in sorted(os.listdir(os.path.join(sim_json_src_dir)))]
num_sim_eval_annots = int(len(sim_annots) * ratio)
print(f"# of eval data in sim: {num_sim_eval_annots}")

# Move trainset
for annot in sim_annots[:-num_sim_eval_annots]:
    # Move json
    data_path = os.path.join(sim_json_src_dir, f"{annot}.json")
    shutil.move(data_path, os.path.join(sim_dst_dir, "train", "dataset_json"))
    # Move numpy
    data_path = os.path.join(sim_numpy_src_dir, f"{annot}.npz")
    shutil.move(data_path, os.path.join(sim_dst_dir, "train", "dataset_numpy"))

# Move evalset
for annot in sim_annots[-num_sim_eval_annots:]:
    # Move json
    data_path = os.path.join(sim_json_src_dir, f"{annot}.json")
    shutil.move(data_path, os.path.join(sim_dst_dir, "eval", "dataset_json"))
    # Move numpy
    data_path = os.path.join(sim_numpy_src_dir, f"{annot}.npz")
    shutil.move(data_path, os.path.join(sim_dst_dir, "eval", "dataset_numpy"))
