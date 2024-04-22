import open3d as o3d
import numpy as np
from typing import List, Union

from imm.pybullet_util.typing_extra import ( TranslationT, Tuple3 )
from imm.pybullet_util.bullet_client import BulletClient

from utils.process_geometry import ( matrix_visual_to_com,
                                     matrix_com_to_base, 
                                     matrix_base_to_world, 
                                     matrix_camera_to_world )


def calculate_mesh_iou_between_pair(bc:BulletClient,
                                    uid_list_A:List[int],
                                    uid_list_B:List[int],
                                    reduction:str="mean",
                                    num_points:int=32768) -> Union[float, List[float]]:
    """Calculate mesh iou of given pair distinguished by the index (using open3d backend)
    Args:
        bc (BulletClient): bullet client id
        uidA (int): pybullet URDF instance id
        uidB (int): pybullet URDF instance id
        reduction (str): mean or none
        num_points (int): number of points for IoU calculation
    
    Returns:
        float|List[float]: calculated IoU.
    """

    iou_list = []
    for uidA, uidB in zip(uid_list_A, uid_list_B):
        # Transform the mesh to the target coordinate
        T_v2l_A = matrix_visual_to_com(bc, uidA)
        T_l2b_A = matrix_com_to_base(bc, uidA)
        T_b2w_A = matrix_base_to_world(bc, uidA)
        T_v2w_A = np.matmul(T_b2w_A,
                            np.matmul(T_l2b_A, T_v2l_A))
        base_link_info_A = bc.getVisualShapeData(uidA)[0]  # [0] is base link
        mesh_A = o3d.io.read_triangle_mesh(base_link_info_A[4].decode('UTF-8'))
        mesh_A.transform(T_v2w_A)

        T_v2l_B = matrix_visual_to_com(bc, uidB)
        T_l2b_B = matrix_com_to_base(bc, uidB)
        T_b2w_B = matrix_base_to_world(bc, uidB)
        T_v2w_B = np.matmul(T_b2w_B,
                            np.matmul(T_l2b_B, T_v2l_B))
        base_link_info_B = bc.getVisualShapeData(uidB)[0]  # [0] is base link
        mesh_B = o3d.io.read_triangle_mesh(base_link_info_B[4].decode('UTF-8'))
        mesh_B.transform(T_v2w_B)


        # Sample query points
        vertices_A = np.asarray(mesh_A.vertices)
        vertices_B = np.asarray(mesh_B.vertices)
        vertices_all = np.concatenate((vertices_A, vertices_B), axis=0)
        box_max = np.max(vertices_all, axis=0)
        box_min = np.min(vertices_all, axis=0)    
        box_center = (box_max + box_min) / 2.
        box_size = (box_max - box_min)
        unit_uniform = np.random.uniform(0, 1, size=(num_points, 3))

        query_points = (unit_uniform-0.5)*box_size + box_center


        # Occupancy query
        query_points_t = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)

        mesh_t_A = o3d.t.geometry.TriangleMesh.from_legacy(mesh_A)    
        scene_A = o3d.t.geometry.RaycastingScene()
        _ = scene_A.add_triangles(mesh_t_A)
        occupancy_A = scene_A.compute_occupancy(query_points_t).numpy()

        mesh_t_B = o3d.t.geometry.TriangleMesh.from_legacy(mesh_B)
        scene_B = o3d.t.geometry.RaycastingScene()
        _ = scene_B.add_triangles(mesh_t_B)
        occupancy_B = scene_B.compute_occupancy(query_points_t).numpy()

        # IoU calculation
        num_intersection = np.sum(np.logical_and(occupancy_A, occupancy_B))
        num_union = np.sum(np.logical_or(occupancy_A, occupancy_B))
        iou = float(num_intersection)/float(num_union)

        iou_list.append(iou)


    # Return
    if reduction=="mean":
        return sum(iou_list)/float(len(iou_list))
    if reduction==None or reduction=="none":
        return iou_list




def calculate_mesh_iou_between_environment(bc:BulletClient,
                                           uid_list_A:List[int],
                                           uid_list_B:List[int],
                                           num_points:int=262144) -> float:
    """Calculate mesh iou between two environments (using open3d backend)

    Args:
        bc (BulletClient): bullet client id
        uid_list_A (List[int]): list of pybullet URDF instance id
        uid_list_B (List[int]): list of pybullet URDF instance id
        num_points (int): number of points for IoU calculation
    
    Returns:
        float: calculated IoU
    """

    # Transform the mesh to the target coordinate
    mesh_list_A = []
    for uid in uid_list_A:
        T_v2l = matrix_visual_to_com(bc, uid)
        T_l2b = matrix_com_to_base(bc, uid)
        T_b2w = matrix_base_to_world(bc, uid)
        T_v2w = np.matmul(T_b2w,
                          np.matmul(T_l2b, T_v2l))
        base_link_info = bc.getVisualShapeData(uid)[0]  # [0] is base link
        mesh = o3d.io.read_triangle_mesh(base_link_info[4].decode('UTF-8'))
        mesh.transform(T_v2w)
        mesh_list_A.append(mesh)

    mesh_list_B = []
    for uid in uid_list_B:
        T_v2l = matrix_visual_to_com(bc, uid)
        T_l2b = matrix_com_to_base(bc, uid)
        T_b2w = matrix_base_to_world(bc, uid)
        T_v2w = np.matmul(T_b2w,
                          np.matmul(T_l2b, T_v2l))
        base_link_info = bc.getVisualShapeData(uid)[0]  # [0] is base link
        mesh = o3d.io.read_triangle_mesh(base_link_info[4].decode('UTF-8'))
        mesh.transform(T_v2w)
        mesh_list_B.append(mesh)

    # Sample query points
    vertices_all = np.zeros((0, 3), np.float64)
    print(vertices_all.shape)
    for mesh in mesh_list_A:
        vertices_new = np.asarray(mesh.vertices)
        vertices_all = np.concatenate((vertices_all, vertices_new), axis=0)
    for mesh in mesh_list_B:
        vertices_new = np.asarray(mesh.vertices)
        vertices_all = np.concatenate((vertices_all, vertices_new), axis=0)
    box_max = np.max(vertices_all, axis=0)
    box_min = np.min(vertices_all, axis=0)    
    box_center = (box_max + box_min) / 2.
    box_size = (box_max - box_min)
    unit_uniform = np.random.uniform(0, 1, size=(num_points, 3))

    query_points = (unit_uniform-0.5)*box_size + box_center

    # Occupancy query
    query_points_t = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)


    occupancy_all_A = np.zeros(num_points, dtype=np.uint)
    for mesh in mesh_list_A:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)    
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        occupancy_new = scene.compute_occupancy(query_points_t).numpy()
        occupancy_all_A = np.logical_or(occupancy_all_A, occupancy_new)

    occupancy_all_B = np.zeros(num_points, dtype=np.uint)
    for mesh in mesh_list_B:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)    
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        occupancy_new = scene.compute_occupancy(query_points_t).numpy()
        occupancy_all_B = np.logical_or(occupancy_all_B, occupancy_new)

    # IoU calculation
    num_intersection = np.sum(np.logical_and(occupancy_all_A, occupancy_all_B))
    num_union = np.sum(np.logical_or(occupancy_all_A, occupancy_all_B))
    iou = float(num_intersection)/float(num_union)


    return iou



def __test_main():
    """Main for testing mesh IoU..."""
    import pybullet as pb
    import pybullet_data
    from pybullet_object_models import ycb_objects
    import matplotlib.pyplot as plt
    import os

    sim_id = pb.connect(pb.GUI)
    bc = BulletClient(sim_id)

    plane_uid = bc.loadURDF(
        fileName        = os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 
        basePosition    = (0.0, 0.0, 0.0), 
        baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        useFixedBase    = True)

    uid_1_A = bc.loadURDF(
        fileName        = os.path.join(ycb_objects.getDataPath(), "YcbBanana", "model.urdf"),
        basePosition    = [0.0, 0.1, 0],
        baseOrientation = bc.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase    = False)

    uid_2_A = bc.loadURDF(
        fileName        = os.path.join(ycb_objects.getDataPath(), "YcbHammer", "model.urdf"),
        basePosition    = [0.5, 0, 0],
        baseOrientation = bc.getQuaternionFromEuler([0, 0, 1.57]),
        useFixedBase    = False)
    
    uid_1_B = bc.loadURDF(
        fileName        = os.path.join(ycb_objects.getDataPath(), "YcbBanana", "model.urdf"),
        basePosition    = [0., 0.0, 0],
        baseOrientation = bc.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase    = False)

    uid_2_B = bc.loadURDF(
        fileName        = os.path.join(ycb_objects.getDataPath(), "YcbHammer", "model.urdf"),
        basePosition    = [0.8, 0.0, 0],
        baseOrientation = bc.getQuaternionFromEuler([0, 0, 1.57]),
        useFixedBase    = False)



    print(f"IoU {calculate_mesh_iou_between_pair(bc, [uid_1_A, uid_2_A], [uid_1_B, uid_2_B])}")

    while True:
        pass
    




if __name__=="__main__":
    __test_main()

