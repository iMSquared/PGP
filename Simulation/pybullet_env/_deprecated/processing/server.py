import multiprocessing as mp
from typing import Dict 

from utils.process_pointcloud import visualize_point_cloud

from processing.msgs import (
    FetchingRequestMsg, FetchingResponseMsg,
    ActionInfo, ObservationInfo
)


class FetchingTaskServer:
    '''
    A callable class for server process
    '''

    def __init__(self, config:Dict, name:str):
        '''
        Set configurations.
        This default constructor does not actually initialize anything.
        '''
        # Server name
        self.name = name
        self.config = config



    def __init_task_server(self):
        '''
        A private attribute that initialize the task within the __call__() of each process.
        '''
        # A trick for launching pybullet in each thread.
        import pybullet as pb
        from imm.pybullet_util.bullet_client import BulletClient
        from envs.binpick_env import BinpickEnv
        from envs.robot import UR5, FrankaPanda
        from envs.manipulation import Action
        from envs.grasp_pose_sampler import PointSamplingGrasp
        from observation.observation import DistanceModel


        # Connect bullet client
        if self.config["project_params"]["debug"]["show_gui"]:
            sim_id = pb.connect(pb.GUI)
        else:
            sim_id = pb.connect(pb.DIRECT)
        if sim_id < 0:
            raise ValueError("Failed to connect to pybullet!")
        self.bc = BulletClient(sim_id)
        #self.bc.setPhysicsEngineParameter(enableFileCaching=0)   # Turn off file caching


        # Use external GPU for rendering when available
        if self.config["project_params"]["use_nvidia"] == True:
            import pkgutil
            egl = pkgutil.get_loader('eglRenderer')
            if (egl):
                eglPluginId = self.bc.loadPlugin(egl.get_filename(), "_eglRendererPlugin")


        # Sim params
        CONTROL_DT = 1. / self.config["sim_params"]["control_hz"]
        self.bc.setTimeStep(CONTROL_DT)
        self.bc.setGravity(0, 0, self.config["sim_params"]["gravity"])
        self.bc.resetDebugVisualizerCamera(
            cameraDistance       = self.config["sim_params"]["debug_camera"]["distance"], 
            cameraYaw            = self.config["sim_params"]["debug_camera"]["yaw"], 
            cameraPitch          = self.config["sim_params"]["debug_camera"]["pitch"], 
            cameraTargetPosition = self.config["sim_params"]["debug_camera"]["target_position"])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING)


        # Init environments, agents and action
        # env
        self.binpick_env = BinpickEnv(self.bc, self.config)
        # robot
        if self.config["project_params"]["robot"] == "ur5":
            self.robot = UR5(self.bc, self.config)
        elif self.config["project_params"]["robot"] == "franka_panda":
            self.robot = FrankaPanda(self.bc, self.config)
        else:
            raise Exception("invalid robot")
        # action
        self.action = Action(self.bc, self.robot, self.config)
        # observation
        self.observation_model = DistanceModel(self.bc, self.config)
        
        # grasp pose sampler
        self.sample_grasp_pose = PointSamplingGrasp()



        # Config process
        self.request_timeout = self.config["project_params"]["processing"]["request_timeout"]
        print(f"Log: Server {self.name} ready")



    def __call__(
            self, 
            request_queue: mp.Queue, 
            response_queue: mp.Queue):
        '''
        The callable attribute for each process.
        '''

        # Launch pybullet
        self.__init_task_server()

        # Stabilize the environment
        self.action.wait(100)

        state_pcd = self.binpick_env.get_pcd()


        # Keep processing the requests 
        while True:

            # Wait the resquest from the client (Synchronous blocking call)
            req:FetchingRequestMsg = request_queue.get(
                block   = True, 
                timeout = self.request_timeout)
            
            # TODO: Reset to initial state from the request

            # Pick demo
            grasp_pos, grasp_orn_q = self.sample_grasp_pose(state_pcd[self.binpick_env.objects_uid[0]]) # A point on the surface.
            self.action.pick(grasp_pos, grasp_orn_q)

            # Action request demo
            action_info = req.action_info
            next_action = getattr(self.action, action_info.action)
            next_action(action_info.pos, action_info.orn)

            # Stabilize
            self.action.wait(100)

            # Observation demo
            measurement = self.binpick_env.get_measurement()
            state_pcd = self.binpick_env.get_pcd()
            likelihood, maplines = self.observation_model(measurement, state_pcd, average_by_segment=False)

            # Write response
            res = FetchingResponseMsg(
                result_state_info = None,               # TODO
                observation_info  = ObservationInfo(    
                    observation = None,                 # TODO
                    likelihood = likelihood))
            try:
                response_queue.put(res, block=False)
            except:
                print(f"res_{self.name}: Invalid synchronous call")
                raise ValueError

            # Visualize 
            #visualize_point_cloud([state, measurement], maplines)

            


