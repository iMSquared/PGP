import subprocess
import time


num_process = 32

# Execute childs
processes = []
for i in range(num_process):
    proc = subprocess.Popen('python ./vessl_fetching_problem_primitive_object.py --save_path="./output" --num_executions=10000', shell=True, stdout=subprocess.DEVNULL)
    processes.append(proc)

    time.sleep(0.1) 

# Block wait
print("All child processes executed. Waiting...")
for i in range(num_process):
    processes[i].communicate()
