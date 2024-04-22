import argparse
import os
import yaml

import multiprocessing as mp 

from processing.server import FetchingTaskServer
from processing.client import FetchingTaskClient

def main(config):
    '''
    Program main (entry point)
    '''

    # Define the function for the `planner` process
    simplan_task_server = FetchingTaskServer(config, "simplan")
    request_queue_simplan = mp.Queue(maxsize=1)
    response_queue_simplan = mp.Queue(maxsize=1)

    # Define the function for the `executor` process
    simexec_task_server = FetchingTaskServer(config, "simexec")
    request_queue_simexec = mp.Queue(maxsize=1)
    response_queue_simexec = mp.Queue(maxsize=1)

    # Define the process that runs the planner
    planner_task_client = FetchingTaskClient(config, "planner")
    
    # Init processses
    process_planner_client = mp.Process(
        target = planner_task_client, 
        args = (
            request_queue_simplan,
            response_queue_simplan,
            request_queue_simexec,
            response_queue_simexec))

    process_simplan_server = mp.Process(
        target = simplan_task_server,
        args = (
            request_queue_simplan,
            response_queue_simplan)) 

    process_simexec_server = mp.Process(
        target = simexec_task_server,
        args = (
            request_queue_simexec,
            response_queue_simexec))

    print("main: processes ready")

    # Start processes
    process_planner_client.start()
    process_simplan_server.start()
    process_simexec_server.start()

    print("main: all processes are alive")

    # Check process healthy
    while True:
        if not (process_planner_client.is_alive() \
            and process_simplan_server.is_alive() \
            and process_simexec_server.is_alive()):
            
            # Terminate all processes
            print("main: some process has died ")
            process_planner_client.terminate()
            process_simplan_server.terminate()
            process_simexec_server.terminate()

            break

    print("main: terminating the program")

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)