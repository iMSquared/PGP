import subprocess
import time


num_process = 10
num_episodes_per_process = 10
num_sims = 100

# Execute childs
processes = []
for i in range(num_process):
    proc = subprocess.Popen(f'python ./fetching_problem_primitive_object.py \
                            --num_episodes={num_episodes_per_process} \
                            --override_num_sim={num_sims} \
                            --override_collect_data\
                            --override_sftp \
                            --override_dataset_save_path="./vessl_output" \
                            --override_dataset_save_path_sftp="/home/ssd2/sanghyeon/vessl/May16th_3obj_newpick_sparse_1000"', 
                            shell=True, stdout=subprocess.DEVNULL)
    processes.append(proc)

    time.sleep(2.0) # Suggest to wait OVER the 1 second.

# Block wait
print("All child processes executed. Waiting...")
for i in range(num_process):
    processes[i].communicate()
