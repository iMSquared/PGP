import pickle
import os
import math
from typing import Dict, Tuple, List, Iterable
import numpy as np
import numpy.typing as nptype
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
import numpy.typing as npt

import pybullet_data


from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, EulerT
from envs.global_object_id_table import GlobalObjectIDTable, Gid_T

from utils.process_pointcloud import DepthToPointcloudOpenGL
from utils.process_geometry import random_sample_array_from_config



class BinpickEnvPrimitive:

    def __init__(self, bc, config):
        self.bc = bc
        self.config = config
        env_params = config["env_params"]["binpick_env"]

        # Configs
        DEBUG_SHOW_GUI      : bool                       = config["project_params"]["debug"]["show_gui"]
        CUSTOM_URDF_DIR_PATH: str                        = config["project_params"]["custom_urdf_path"]
        CABINET_PATH        : str                        = env_params["cabinet"]["path"]
        CABINET_POS         : TranslationT               = env_params["cabinet"]["pos"]
        CABINET_ORN         : EulerT                     = env_params["cabinet"]["orn"]
        GOAL_POS            : TranslationT               = env_params["goal"]["pos"]
        GOAL_COLOR          : Tuple[float, float, float] = env_params["goal"]["color"]
        OBJECT_CONFIGS      : List[Dict]                 = env_params["objects"]

        self.TASKSPACE_CENTER                     : TranslationT = env_params["taskspace"]["center"]
        self.TASKSPACE_HALF_RANGES                : TranslationT = env_params["taskspace"]["half_ranges"]
        self.RANDOM_INIT_TARGET_POS_CENTER        : TranslationT = env_params["random_init"]["target"]["pos_center"]
        self.RANDOM_INIT_TARGET_POS_HALF_RANGES   : TranslationT = env_params["random_init"]["target"]["pos_half_ranges"]
        self.RANDOM_INIT_TARGET_ORN_CENTER        : EulerT       = env_params["random_init"]["target"]["orn_center"]
        self.RANDOM_INIT_TARGET_ORN_HALF_RANGES   : EulerT       = env_params["random_init"]["target"]["orn_half_ranges"]
        self.RANDOM_INIT_NONTARGET_POS_CENTER     : TranslationT = env_params["random_init"]["nontarget"]["pos_center"]
        self.RANDOM_INIT_NONTARGET_POS_HALF_RANGES: TranslationT = env_params["random_init"]["nontarget"]["pos_half_ranges"]
        self.RANDOM_INIT_NONTARGET_ORN_CENTER     : EulerT       = env_params["random_init"]["nontarget"]["orn_center"]
        self.RANDOM_INIT_NONTARGET_ORN_HALF_RANGES: EulerT       = env_params["random_init"]["nontarget"]["orn_half_ranges"]

        # Path to URDF
        pybullet_data_path = pybullet_data.getDataPath()
        file_path = Path(__file__)
        project_path = file_path.parent.parent
        urdf_dir_path = os.path.join(project_path, CUSTOM_URDF_DIR_PATH)


        # Load environmental URDFs
        self.plane_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "plane.urdf"), 
            basePosition    = (0.0, 0.0, 0.0), 
            baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase    = True)
        self.cabinet_uid = self.bc.loadURDF(
            fileName        = os.path.join(urdf_dir_path, CABINET_PATH),
            basePosition    = CABINET_POS,
            baseOrientation = self.bc.getQuaternionFromEuler(CABINET_ORN),
            useFixedBase    = True)
        # Create table for goal
        self.vis_box_shape_id = self.bc.createVisualShape(
            shapeType = self.bc.GEOM_BOX,
            halfExtents = [0.1, 0.1, 0.295],
            rgbaColor = [0.5, 0.3, 0.2, 1.0])
        self.col_box_shape_id = self.bc.createCollisionShape(
            shapeType = self.bc.GEOM_BOX,
            halfExtents = [0.1, 0.1, 0.295])
        self.goal_table = self.bc.createMultiBody(
            baseMass = 1000,
            baseCollisionShapeIndex = self.col_box_shape_id,
            baseVisualShapeIndex = self.vis_box_shape_id,
            basePosition = GOAL_POS)
        self.bc.createConstraint(parentBodyUniqueId = self.plane_uid, 
                                 parentLinkIndex = -1, 
                                 childBodyUniqueId = self.goal_table,
                                 childLinkIndex = -1, 
                                 jointType = self.bc.JOINT_FIXED, 
                                 jointAxis = [0, 0, 0], 
                                 parentFramePosition = GOAL_POS, 
                                 childFramePosition = [0, 0, 0])


        # Show task space region when debugging
        if DEBUG_SHOW_GUI:
            self.draw_taskpace()
        

        # Load objects URDF
        self.object_uids: List[int]           = []
        self.gid_to_uid : dict[Gid_T, int]    = dict()
        self.uid_to_gid : dict[int, Gid_T]    = dict()
        self.gid_table  : GlobalObjectIDTable = GlobalObjectIDTable()
        # Just give gid from 0 to 1, 2, ...
        for i, obj_config in enumerate(OBJECT_CONFIGS):
            # Read configs
            pos            = obj_config["pos"]
            orn            = obj_config["orn"]
            is_target      = obj_config["is_target"]
            urdf_file_path = os.path.join(urdf_dir_path, obj_config["urdf_file_path"])
            pcd_file_path  = os.path.join(urdf_dir_path, obj_config["pcd_file_path"])
            # Load urdf and pointcloud
            uid = self.bc.loadURDF(fileName = urdf_file_path,
                                   basePosition = pos,
                                   baseOrientation = self.bc.getQuaternionFromEuler(orn),
                                   useFixedBase = False)
            with open(pcd_file_path, "rb") as f:
                pointcloud: Dict[str, Tuple[TranslationT]] = pickle.load(f)
            pcd_points = np.asarray(pointcloud["points"])
            pcd_normals = np.asarray(pointcloud["normals"])

            # Save uid and gid
            shape_info = GlobalObjectIDTable.ShapeInfoEntry(
                urdf_file_path = urdf_file_path,
                pcd_file_path = pcd_file_path,
                pcd_points = pcd_points,
                pcd_normals = pcd_normals)
            gid_header = GlobalObjectIDTable.Header(
                is_target = is_target,
                shape_info = shape_info)
            
            # Assigning gid
            gid: Gid_T = i
            self.object_uids.append(uid)
            self.gid_to_uid[gid] = uid
            self.uid_to_gid[uid] = gid
            self.gid_table[gid] = gid_header


        # Adjust dynamics
        self.reset_dynamics()


        # RGB-D camera config (intrinsic)
        camera_params = env_params["depth_camera"]

        self.WIDTH     = camera_params["width"]
        self.HEIGHT    = camera_params["height"]
        CAM_FOV        = camera_params["fov"]
        CAM_TARGET_POS = camera_params["target_pos"]
        CAM_DISTANCE   = camera_params["distance"]
        CAM_ROLL       = camera_params["roll"]
        CAM_PITCH      = camera_params["pitch"]
        CAM_YAW        = camera_params["yaw"]
        CAM_NEAR_VAL   = camera_params["near_val"]
        CAM_FAR_VAL    = camera_params["far_val"]

        self.CAMERA_VIEW_MATRIX = self.bc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition = CAM_TARGET_POS,
            distance             = CAM_DISTANCE,
            roll                 = CAM_ROLL,
            pitch                = CAM_PITCH,
            yaw                  = CAM_YAW,
            upAxisIndex          = 2)
        
        self.CAMERA_PROJ_MATRIX = self.bc.computeProjectionMatrixFOV(
            fov     = CAM_FOV,
            aspect  = float(self.WIDTH) / float(self.HEIGHT),
            nearVal = CAM_NEAR_VAL,
            farVal  = CAM_FAR_VAL)

        self.reprojection_function = DepthToPointcloudOpenGL(
            cx                     = self.WIDTH / 2.0,
            cy                     = self.HEIGHT / 2.0,
            near_val               = CAM_NEAR_VAL,
            far_val                = CAM_FAR_VAL,
            focal_length           = self.HEIGHT / ( 2.0 * math.tan(math.radians(CAM_FOV) / 2.0) ),)
            # voxel_down_sample_size = camera_params["voxel_down_sample_size"])

        # NOTE(ssh):For the calculatation of the focal length in pybullet, see the material below
        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet



    def reset_object_poses_to_random(self, check_full_occlusion=True):
        """Randomly reset objects in binpick environment.
        Make sure to call this function after the environment is stabilized.

        Args:
            check_full_occlusion (bool): Fully occluded the target object when True.
        """
        while True:
            # Shuffle non-target objects
            for gid in self.gid_to_uid.keys():                  
                uid = self.gid_to_uid[gid]
                base_pos, base_orn_q = self.bc.getBasePositionAndOrientation(uid)
                if self.gid_table[gid].is_target:
                    # Noising the target object
                    random_pos = random_sample_array_from_config(center      = self.RANDOM_INIT_TARGET_POS_CENTER,
                                                                 half_ranges = self.RANDOM_INIT_TARGET_POS_HALF_RANGES)
                    random_orn = random_sample_array_from_config(center      = self.RANDOM_INIT_TARGET_ORN_CENTER,
                                                                 half_ranges = self.RANDOM_INIT_TARGET_ORN_HALF_RANGES)
                    random_pos[2] = base_pos[2]
                    self.bc.resetBasePositionAndOrientation(
                        uid, random_pos, self.bc.getQuaternionFromEuler(random_orn))
                else:
                    # Noising the non-target object
                    random_pos = random_sample_array_from_config(center      = self.RANDOM_INIT_NONTARGET_POS_CENTER,
                                                                 half_ranges = self.RANDOM_INIT_NONTARGET_POS_HALF_RANGES)
                    random_orn = random_sample_array_from_config(center      = self.RANDOM_INIT_NONTARGET_ORN_CENTER,
                                                                 half_ranges = self.RANDOM_INIT_NONTARGET_ORN_HALF_RANGES)
                    random_pos[2] = base_pos[2]
                    self.bc.resetBasePositionAndOrientation(
                        uid, random_pos, self.bc.getQuaternionFromEuler(random_orn))
            # Check collision
            has_contact = self.check_objects_close()
            # Check target object occlusion
            _, _, seg_mask = self.capture_rgbd_image(sigma=0)
            target_gid = self.gid_table.select_target_gid()
            is_target_in_obs: bool = np.array(np.sum(seg_mask[target_gid]), dtype=bool).item()
            # End randomization when no collision exist
            if check_full_occlusion:
                if not has_contact and not is_target_in_obs:
                    break
            else:
                if not has_contact:
                    break                
            print("Reinit ground truth due to the collision")                


    def check_objects_close(self, closest_threshold: float = 0.01) -> bool:
        """Check objects are close within the threshold.

        Args:
            closest_threshold (float, optional): Defaults to 0.02.
        """
        # Query closest points
        is_close = False
        for uid_a in self.object_uids:
            for uid_b in self.object_uids:
                if uid_a == uid_b:
                    continue
                # Set flag
                closest_points = self.bc.getClosestPoints(uid_a, uid_b, closest_threshold, -1, -1,)
                if len(closest_points) > 0:
                    is_close = True
                    break

        return is_close


    def check_objects_collision(self) -> bool:
        """Check objects have collision in between
        NOTE(ssh): use check_objects_close instead.

        Returns:
            bool: True when collision exists
        """
        # Check collision
        has_contact = False
        self.bc.performCollisionDetection()
        for uid_a in self.object_uids:
            for uid_b in self.object_uids:
                if uid_a == uid_b:
                    continue
                # Set flag
                contact_points = self.bc.getContactPoints(uid_a, uid_b, -1, -1)
                if len(contact_points) > 0:
                    has_contact = True
                    break

        return has_contact


    def capture_rgbd_image(self, sigma) \
            -> Tuple[ nptype.NDArray, nptype.NDArray, Dict[Gid_T, nptype.NDArray] ]:
        """Capture_rgbd_image

        Returns:
            depth_array (NDArray): Pixel depth value [H, W]
            rgb_array (NDArray): Pixel RGB value [H, W, 3] ranges within [0, 255].
            seg_mask (Dict[Gid_T, NDArray]): Key is GID.
        """
        (w, h, px, px_d, px_id) = self.bc.getCameraImage(
            width            = self.WIDTH,
            height           = self.HEIGHT,
            viewMatrix       = self.CAMERA_VIEW_MATRIX,
            projectionMatrix = self.CAMERA_PROJ_MATRIX,
            renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)

        # Reshape list into ndarray(image)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]                 # remove alpha

        depth_array = np.array(px_d, dtype=np.float32)
        noise = sigma * np.random.randn(h, w)
        depth_array = depth_array + noise
        
        # Segmentation
        mask_array = np.array(px_id, dtype=np.uint8)
        seg_mask = {}
        for gid, uid in self.gid_to_uid.items():
            seg_mask[gid] = np.where(mask_array==uid, True, False)

        return depth_array, rgb_array, seg_mask
    

    def reset_dynamics(self):
        """Reset the dynamics of spawned objects."""
        for uid in self.object_uids:
            self.bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=0.8,
                rollingFriction=0.0004,
                spinningFriction=0.0004,
                restitution=0.2)
            

    def draw_taskpace(self):
        """Draw task space region for debugging"""

        center = self.TASKSPACE_CENTER
        half_ranges = self.TASKSPACE_HALF_RANGES

        point1 = np.array(center)
        point1[0] += half_ranges[0]
        point1[1] += half_ranges[1]

        point2 = np.array(center)
        point2[0] += half_ranges[0]
        point2[1] -= half_ranges[1]

        point3 = np.array(center)
        point3[0] -= half_ranges[0]
        point3[1] -= half_ranges[1]

        point4 = np.array(center)
        point4[0] -= half_ranges[0]
        point4[1] += half_ranges[1]

        self.bc.addUserDebugLine(point1, point2, [0, 0, 1])
        self.bc.addUserDebugLine(point2, point3, [0, 0, 1])
        self.bc.addUserDebugLine(point3, point4, [0, 0, 1])
        self.bc.addUserDebugLine(point4, point1, [0, 0, 1])  