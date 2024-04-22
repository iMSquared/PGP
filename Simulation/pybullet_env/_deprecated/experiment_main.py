import subprocess
import time



num_sims_to_plot = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
num_process = 50

for num_sims in num_sims_to_plot:
    
    print(f"experiment with paricles={num_sims}")

    # Execute childs
    processes = []
    for i in range(num_process):
        proc = subprocess.Popen(f"python ./fetching_problem_primitive_object.py --num_executions=1 --policy=random --num_sims={num_sims}", shell=True, stdout=subprocess.DEVNULL)
        processes.append(proc)

        time.sleep(0.1) 

    # Block wait
    time.sleep(0.01)
    print("All child processes executed. Waiting...")
    for i in range(num_process):
        processes[i].communicate()

    

