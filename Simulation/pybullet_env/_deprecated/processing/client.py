import math
import random
import multiprocessing as mp
from typing import Dict 

from processing.msgs import (
    FetchingRequestMsg, FetchingResponseMsg,
    ActionInfo, ObservationInfo
)



class FetchingTaskClient:
    '''
    A callable class for client process
    '''

    def __init__(self, config:Dict, name:str):
        '''
        Set configurations.
        This default constructor does not actually initialize anything.
        '''
        # Client name
        self.name = name
        self.config = config
        


    def __init_task_client(self):
        '''
        A private attribute that initialize the task within the __call__() of each process.
        '''
        # Do whatever needed here.


        # Config process
        self.request_timeout = self.config["project_params"]["processing"]["request_timeout"]
        print(f"Log: Server {self.name} ready")


    def __call__(
            self, 
            request_queue_planner: mp.Queue, 
            response_queue_planner: mp.Queue):
        '''
        The callable attribute for each process.
        '''

        # Launch POMCPOW or whatever
        self.__init_task_client()

        action_poses = (
            (0.2, 0.0, 0.7),
            (0.3, 0.0, 0.7),
            (0.4, 0.0, 0.7),
            (0.5, 0.0, 0.7),)


        # Keep processing the requests 
        while True:
            # Test
            pos = action_poses[random.randint(0,3)]
            
            # Write request message
            print(f"In {self.name}: sending request to simplan")
            req = FetchingRequestMsg(
                initial_state_info = None,
                action_info        = ActionInfo(action = "move",
                                                pos    = pos,
                                                orn    = (0, 0.9236508, 0, 0.3832351)))
            # Send action request to planner
            try:
                request_queue_planner.put(req, block=False)
            except:
                print(f"req_{self.name}: Invalid synchronous call")
                raise ValueError

            # Action is now being executed...


            # Wait the response from planner (Synchronous blocking call)
            print(f"In {self.name}: waiting response from simplan")
            try:
                res:FetchingResponseMsg = response_queue_planner.get(
                    block   = True, 
                    timeout = self.request_timeout)
            except:
                print(f"res_{self.name}: Timeout error. Terminating the process")
                raise ValueError


            # Test
            print(f"In {self.name}: simplan likelihood = {res.observation_info.likelihood}")
