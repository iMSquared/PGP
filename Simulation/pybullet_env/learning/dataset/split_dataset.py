import os
import shutil


src_dir = "/home/share_folder/dataset/4.11/1000/sim/fail"
dst_dir = "/home/share_folder/dataset/4.11/1000/sim/fail_eval"
ratio = 0.2

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

dataset = os.listdir(src_dir)
num_data_move = int(len(dataset) * ratio)
print(num_data_move)

for d in dataset[-num_data_move:]:
    data_path = os.path.join(src_dir, d)
    shutil.move(data_path, dst_dir)
    # shutil.copy(data_path, dst_dir)