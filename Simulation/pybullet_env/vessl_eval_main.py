import subprocess
import time
import argparse
import zipfile
import glob
import os
from data_generation.collect_data import time_stamp



def get_args():
    """Parse experiment configuration"""
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--num_processes",                   type=int, default=1, help="Number of processes to run for experiments")
    parser.add_argument("--config",                          type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    parser.add_argument("--num_episodes",                    type=int, default=1, help="Number of episodes")
    parser.add_argument("--override_num_sims",               type=int, default=None, help="Number of simulations")
    parser.add_argument("--override_collect_data",           action='store_true', help="Collect data when true")
    parser.add_argument("--override_sftp",                   action='store_true', help="Send data via SFTP")
    parser.add_argument("--override_exp_log_dir_path",       type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_exp_learning_dir_path",  type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_dataset_save_path",      type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_dataset_save_path_sftp", type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_use_guided_policy",      action='store_true', help="Use guided policy when True")
    parser.add_argument("--override_use_guided_value",       action='store_true', help="Use guided value when True")
    parser.add_argument("--override_guide_q_value",          action='store_true', help="Overrides to guide Q value when called")
    parser.add_argument("--override_guide_preference",       action='store_true', help="Overrides to guide preference when called")
    parser.add_argument("--override_inference_device",       type=str, default=None, help="Overrides the default inference device in config.")
    params = parser.parse_args()

    return params



def main(params):
    """asdf"""

    NUM_PROCESSES = params.num_processes
    print(f"num processes    : {params.num_processes}")
    print(f"num simulations  : {params.override_num_sims}")
    print(f"use guided policy: {params.override_use_guided_policy}")
    print(f"use guided value : {params.override_use_guided_value}")
    print(f"guide q value    : {params.override_guide_q_value}")
    print(f"guide preference : {params.override_guide_preference}")
    print(f"inference device : {params.override_inference_device}")

    # Compose process prompt
    prompt = f"python ./fetching_problem_primitive_object.py "
    if params.config is not None:
        prompt += f" --config {params.config} "
    if params.num_episodes is not None:
        prompt += f" --num_episodes {params.num_episodes} "
    if params.override_num_sims is not None:
        prompt += f" --override_num_sims {params.override_num_sims} "
    if params.override_collect_data is True:
        prompt += f" --override_collect_data "
    if params.override_sftp is True:
        prompt += f" --override_sftp "
    if params.override_exp_log_dir_path is not None:
        prompt += f" --override_exp_log_dir_path {params.override_exp_log_dir_path}"
    if params.override_exp_learning_dir_path is not None:
        prompt += f" --override_exp_learning_dir_path {params.override_exp_learning_dir_path}"
    if params.override_dataset_save_path is not None:
        prompt += f" --override_dataset_save_path {params.override_dataset_save_path}"
    if params.override_dataset_save_path_sftp is not None:
        prompt += f" --override_dataset_save_path_sftp {params.override_dataset_save_path_sftp}"
    if params.override_use_guided_policy is True:
        prompt += f" --override_use_guided_policy "
    if params.override_use_guided_value is True:
        prompt += f" --override_use_guided_value "
    if params.override_guide_q_value is True:
        prompt += f" --override_guide_q_value "
    if params.override_guide_preference is True:
        prompt += f" --override_guide_preference "
    if params.override_inference_device is not None:
        prompt += f" --override_inference_device {params.override_inference_device}"
    print(prompt)


    # Execute childs
    processes = []
    for i in range(NUM_PROCESSES):
        # proc = subprocess.Popen(prompt, shell=True, stdout=subprocess.DEVNULL)
        proc = subprocess.Popen(prompt, shell=True)
        processes.append(proc)
        time.sleep(2.0) # Suggest to wait OVER the 1 second.


    # Block wait
    print("All child processes executed. Waiting...")
    for i in range(NUM_PROCESSES):
        processes[i].communicate()


    # Compress and send experiment result
    results = glob.glob(params.override_exp_log_dir_path + '/**', recursive=True)
    with zipfile.ZipFile(os.path.join(params.override_exp_log_dir_path, f'../exp_log_{time_stamp()}.zip'), 'w') as myzip:
        for res in results:
            myzip.write(res)



if __name__ == "__main__":
    args = get_args()
    main(args)


