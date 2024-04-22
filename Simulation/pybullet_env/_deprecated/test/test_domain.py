"""
Example script for motion planning with a robot arm in pybullet.
Adapted from: bullet3/examples/pybullet/gym/pybullet_utils/bullet_client.py
              https://colab.research.google.com/drive/1eXq-Tl3QKzmbXGSKU2hDk0u_EHdfKVd0?usp=sharing#scrollTo=jERNG_5lWOn3
"""

import pybullet as pb
import pybullet_data
import numpy as np
from typing import List
import logging
import time
import math
import functools
import inspect

from fetching_problem import PandaAction


class BulletClient(object):
  """A wrapper for pybullet to manage different clients."""

  def __init__(self, sim_id: int):
    """Creates a Bullet client from an existing simulation.
    Args:
        sim_id: The client connection id.
    """
    self.__sim_id: int = sim_id

  def __getattr__(self, name: str):
    """Inject the client id into Bullet functions."""
    attr = getattr(pb, name)
    if inspect.isbuiltin(attr):
        pb_attr = attr
        attr = functools.partial(attr,
                                 physicsClientId=self.__sim_id)
        # I think this is fine ...
        functools.update_wrapper(attr, pb_attr)
    return attr

  @property
  def sim_id(self) -> int:
      return self.__sim_id


def pick_and_place_without_motion_planner():

    for t in range(1000):
        print(f'\rtimestep {t}...', end='')
        
        target_pos, gripper_val = [0.72, -0.2, 0.28], 0
        if t >= 150 and t < 250:
            target_pos, gripper_val = [0.72, -0.2, 0.28], 1 # grab object
        elif t >= 250 and t < 400:
            target_pos, gripper_val = [0.72, -0.2, 0.28 + 0.53*(t-250)/150.], 1 # move up after picking object
        elif t >= 400 and t < 600:
            target_pos, gripper_val = [0.72, -0.2 - 0.4*(t-400)/200., 0.85], 1 # move to target position
        elif t >= 600 and t < 700:
            target_pos, gripper_val = [0.72, -0.6, 0.85], 1 # stop at target position
        elif t >= 700:
            target_pos, gripper_val = [0.72, -0.6, 0.85], 0 # drop object

        target_orn = bc.getQuaternionFromEuler([0, 1.01*np.pi, 0])
        joint_poses = bc.calculateInverseKinematics(robot_id, ee_id, target_pos, target_orn)

        if robot == 'kuka':
            for j in range (num_joints):
                bc.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=bc.POSITION_CONTROL, targetPosition=joint_poses[j])
            bc.setJointMotorControl2(robot_gripper_id, 4, bc.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
            bc.setJointMotorControl2(robot_gripper_id, 6, bc.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
        
        elif robot == 'panda':
            gripper_val = 1 - gripper_val   # open: 1 / close: 0

            for j in range(ee_id + 1):
                bc.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=bc.POSITION_CONTROL, targetPosition=joint_poses[j])
            for j in range(ee_id + 1, num_joints):
                bc.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=bc.POSITION_CONTROL, targetPosition=gripper_val*0.08)

        bc.stepSimulation()
        time.sleep(0.01)


if __name__ == '__main__':
    # Parameters for running this example.
    log_level: str = 'WARN'
    robot: str = 'panda' # 'kuka' or 'panda'
    use_motion_planner: bool = True

    # Configure logging.
    logging.root.setLevel(log_level)
    logging.basicConfig()
    # Connect to pybullet simulator.
    sim_id: int = pb.connect(pb.GUI)
    if sim_id < 0:
        raise ValueError('Failed to connect to simulator!')
    bc = BulletClient(sim_id)

    # Load scene.
    bc.setGravity(0, 0, -9.8)
    bc.resetDebugVisualizerCamera(cameraDistance=0.25, cameraYaw=240, cameraPitch=-40, cameraTargetPosition=[-0.25,0.20,0.8])

    bc.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id: int = bc.loadURDF("table/table.urdf", basePosition=[1.0, -0.2, -0.65], baseOrientation=[0, 0, 0.7071, 0.7071])
    cube_id: int = bc.loadURDF("cube.urdf", basePosition=[0.7, -0.2, 0.0], globalScaling=0.05)
    if robot == 'kuka':
        robot_id = bc.loadURDF("kuka_iiwa/model_vr_limits.urdf", basePosition=[1.4, -0.2, 0.0], baseOrientation=[0.0, 0.0, 0.0, 1.0])
        robot_gripper_id = bc.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]

        # attach gripper to kuka arm
        kuka_cid = bc.createConstraint(robot_id, 6, robot_gripper_id, 0, bc.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0])
        kuka_cid2 = bc.createConstraint(robot_gripper_id, 4, robot_gripper_id, 6, jointType=bc.JOINT_GEAR, jointAxis=[1,1,1], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
        bc.changeConstraint(kuka_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=100)

        # reset robot
        jointPositions = [-0.000000, -0.000000, 0.000000, 1.570793, 0.000000, -1.036725, 0.000001]
        for jointIndex in range(bc.getNumJoints(robot_id)):
            bc.resetJointState(robot_id, jointIndex, jointPositions[jointIndex])
            bc.setJointMotorControl2(robot_id, jointIndex, bc.POSITION_CONTROL, jointPositions[jointIndex], 0)

        # reset gripper 
        bc.resetBasePositionAndOrientation(robot_gripper_id, [0.923103, -0.200000, 1.250036], [-0.000000, 0.964531, -0.000002, -0.263970])
        jointPositions = [0.000000, -0.011130, -0.206421, 0.205143, -0.009999, 0.000000, -0.010055, 0.000000]
        for jointIndex in range(bc.getNumJoints(robot_gripper_id)):
            bc.resetJointState(robot_gripper_id, jointIndex, jointPositions[jointIndex])
            bc.setJointMotorControl2(robot_gripper_id, jointIndex, bc.POSITION_CONTROL, jointPositions[jointIndex], 0)

        ee_id: int = 6  # NOTE(jiyong): hardcoded end-effector joint for kuka robot
        num_joints = bc.getNumJoints(robot_id)
    
    elif robot == 'panda':
        robot_id = bc.loadURDF("franka_panda/panda.urdf", basePosition=[1.4, -0.2, 0.0], baseOrientation=bc.getQuaternionFromEuler([0, 0, math.pi]), useFixedBase=True)
        ee_id: int = 8  # NOTE(jiyong): hardcoded end-effector joint for kuka robot
        num_joints = bc.getNumJoints(robot_id) - 1  # 11

        reset_joint_pos = bc.calculateInverseKinematics(robot_id, 8, [0.6, -0.2, 0.6], [0.0, 0.0, 0.0])  # NOTE(jiyong): hardcoded end-effector joint number for panda robot

        # reset robot
        for i in range(ee_id + 1):
            bc.resetJointState(robot_id, i, reset_joint_pos[i])
            bc.setJointMotorControl2(robot_id, i, bc.POSITION_CONTROL, reset_joint_pos[i], 0)
        for i in range(ee_id + 1, num_joints):
            bc.resetJointState(robot_id, i, 0.08)
            bc.setJointMotorControl2(robot_id, i, bc.POSITION_CONTROL, 0.08, 0)
    
    if use_motion_planner:
        manip = PandaAction(bc, robot_id)
        
        pos = [0.62, -0.21, 0.072]
        orn = bc.getQuaternionFromEuler([0, 1.01*np.pi, 0])
        
        # Show target line and step simulation once
        bc.addUserDebugLine(pos, (0,0,0), (0,1,0), 10, 30.0)
        bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)
        bc.stepSimulation()
        
        manip.pick(pos, orn)
        
        pos[2] = pos[2] + 0.5
        manip.move(pos, orn)
        
        pos[1] = pos[1] - 0.4
        manip.place(pos, orn)
        
        while True:
            manip.stay()
        
    else:
        pick_and_place_without_motion_planner()
        
    time.sleep(2)