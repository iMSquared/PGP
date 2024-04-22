import random
import numpy as np
import open3d as o3d


class PointSamplingGrasp:

    def __init__(self):
        pass

    def __call__(self, pcd):
        '''
        Return grasp pose in SE(3) by following the process as below:
            1) Sample one surface point and obtain its normal vector - it is to be a target pose
            2) Calculate the rotation matrix (1) from default heading axis to the normal vector
            3) Obtain orthonormal vectors to the normal vector
            4) Select one orthonormal vector (align to fingers) and its angle which has the smallest width
            5) Calculate the rotation matrix (2) from default axis aligning to fingers to selected vector in 4)
            6) Sample a pitch angle and calculate its rotation matrix (3)
            7) Calculate target orientation which rotates (1), (2), and (3) sequentially
        '''
        # Select surface point and its normal vector
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5))
        
        pcd_points = np.asarray(pcd.points)
        
        rnd_idx = random.choice(range(len(pcd_points)))
        
        sampled_point = pcd_points[rnd_idx]
        normal_vec = pcd.normals[rnd_idx]
        normal_vec /= np.linalg.norm(normal_vec)

        # Choose the normal vector toward downward
        if normal_vec[2] > 0:
            normal_vec = - normal_vec

        # Calculate rotation matrix from z-axis (default heading axis of RG2) to the normal vector
        z_axis = np.asarray([0, 0, 1])
        tmp_axis = np.cross(z_axis, normal_vec)
        angle_to_normal_vec = np.arccos(np.dot(z_axis, normal_vec))
        R_to_normal_vec = self.rot_matrix_based_normal_axis_and_angle(tmp_axis, angle_to_normal_vec)

        # Obtain orthonormal vectors to the normal vector
        y_axis = np.asarray([0, 1, 0])  # The default axis to align to fingers of RG2
        orth_vec = np.matmul(R_to_normal_vec, y_axis.T).T

        roll_interval = 10*(np.pi/180)
        orth_vecs = self.get_orth_vecs(normal_vec, orth_vec, roll_interval)
        
        # Select one orthonormal vector which has the smallest width. It is the axis to align to fingers after rotating
        width = []
        # thres = 0.5
        for vec in orth_vecs:
            proj_pc, proj_dist = self.proj_to_line(vec, pcd_points, sampled_point)
            # adj_proj_pc = proj_pc[np.where(proj_dist<thres)]
            adj_proj_pc = proj_pc
            dist_adj_proj_pc = np.dot((adj_proj_pc - sampled_point), vec)
            width.append(np.max(dist_adj_proj_pc) - np.min(dist_adj_proj_pc))
        width = np.asarray(width)
        
        # Calculate rotation matrix of rotating wrist
        roll_idx = np.argmin(width)
        roll_delta = roll_interval * roll_idx
        # Prevent rotate too much
        while roll_delta >= np.pi:
            roll_delta -= np.pi
        R_roll1 = self.rot_matrix_based_normal_axis_and_angle(normal_vec, roll_delta)
        R_roll2 = self.rot_matrix_based_normal_axis_and_angle(normal_vec, roll_delta-np.pi)
        
        pitch_axis = orth_vecs[roll_idx]
        pitch_axis /= np.linalg.norm(pitch_axis)
        
        # Sample the pitch angle and calculate its rotation matrix
        pitch = self.sample_pitch()
        R_pitch = self.rot_matrix_based_normal_axis_and_angle(pitch_axis, pitch)

        roll_axis = np.matmul(normal_vec, R_pitch.T)
        roll_axis /= np.linalg.norm(roll_axis)
        
        # Target position
        pos = sampled_point         # RG2 gripper set depth itself when closing
        pos = sampled_point + self.sample_depth() * roll_axis

        #  Target orientation (two possible grasp)
        rot_mat1 = np.matmul(R_roll1, R_to_normal_vec)
        rot_mat1 = np.matmul(R_pitch, rot_mat1)
        orn_q1 = self.rot_matrix_to_quaternion(rot_mat1)
        orn1 = self.euler_from_quaternion(orn_q1)

        rot_mat2 = np.matmul(R_roll2, R_to_normal_vec)
        rot_mat2 = np.matmul(R_pitch, rot_mat2)
        orn_q2 = self.rot_matrix_to_quaternion(rot_mat2)
        orn2 = self.euler_from_quaternion(orn_q2)


        return pos, [orn1, orn2]


    @classmethod
    def rot_matrix_based_normal_axis_and_angle(cls, vec, theta):
        '''
        Return the rotation matrix which means rotating angle based on the normal vector.
        Reference: "Rotation matrix from axix and angle" in https://en.wikipedia.org/wiki/Rotation_matrix
        '''
        vec /= np.linalg.norm(vec) # unit vector
        R = np.asarray([[np.cos(theta) + np.power(vec[0],2) * (1-np.cos(theta)), vec[0]*vec[1]*(1-np.cos(theta)) - vec[2]*np.sin(theta), vec[0]*vec[2]*(1-np.cos(theta)) + vec[1]*np.sin(theta)],
                        [vec[1]*vec[0]*(1-np.cos(theta)) + vec[2]*np.sin(theta), np.cos(theta) + np.power(vec[1],2) * (1-np.cos(theta)), vec[1]*vec[2]*(1-np.cos(theta)) - vec[0]*np.sin(theta)],
                        [vec[2]*vec[0]*(1-np.cos(theta)) - vec[1]*np.sin(theta), vec[2]*vec[1]*(1-np.cos(theta)) + vec[0]*np.sin(theta), np.cos(theta) + np.power(vec[2],2) * (1-np.cos(theta))]])
        return R
    

    @classmethod
    def get_orth_vecs(cls, vec_axis, vec_start, delta):
        '''
        Get orthonormal vectors to `vec_axis`, where the start vector is `vec_start` and the interval rotated angle is delta.
        '''
        R = cls.rot_matrix_based_normal_axis_and_angle(vec_axis, delta)
        orth_vecs = np.empty(shape=(int(np.floor(2*np.pi/delta)), 3))
        orth_vecs[0] = vec_start/np.linalg.norm(vec_start)
        for i in range(1, len(orth_vecs)):
            orth_vec = np.dot(R,orth_vecs[i-1].T).T
            orth_vecs[i] = orth_vec/np.linalg.norm(orth_vec)
        
        return orth_vecs
    

    @classmethod
    def proj_to_line(cls, vec, pc, pos):
        proj_pc = np.matmul(vec.reshape(1,3), pc.T).T * vec / np.linalg.norm(vec) + pos
        proj_dist = np.linalg.norm((pc - proj_pc), axis=1)
        return proj_pc, proj_dist
    

    @classmethod
    def sample_pitch(cls, interval=np.pi/3):
        pitch = (random.random() - 0.5) * interval
        return pitch
    

    @classmethod
    def rot_matrix_to_quaternion(cls, m):    # |NOTE(Jiyong)|: It can be replaced with rotation transform of Scipy
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5/t
            q[0] = (m[2,1] - m[1,2]) * t
            q[1] = (m[0,2] - m[2,0]) * t
            q[2] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t

        return q
    
    @classmethod
    def euler_from_quaternion(cls, orn_q):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)

            Reference: https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
            """
            t0 = +2.0 * (orn_q[3] * orn_q[0] + orn_q[1] * orn_q[2])
            t1 = +1.0 - 2.0 * (orn_q[0] * orn_q[0] + orn_q[1] * orn_q[1])
            roll_x = np.arctan2(t0, t1)
        
            t2 = +2.0 * (orn_q[3] * orn_q[1] - orn_q[2] * orn_q[0])
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = np.arcsin(t2)
        
            t3 = +2.0 * (orn_q[3] * orn_q[2] + orn_q[0] * orn_q[1])
            t4 = +1.0 - 2.0 * (orn_q[1] * orn_q[1] + orn_q[2] * orn_q[2])
            yaw_z = np.arctan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians

    @classmethod
    def sample_depth(cls, scale=0.04): # 0.04 for RG2-gripper
        d = scale * random.random()
        return d



