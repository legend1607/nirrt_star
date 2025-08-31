import os
import numpy as np
import pybullet as p
import pybullet_data
from time import sleep, time

class KukaEnv:
    """
    Comprehensive KUKA environment class for motion planning and dataset generation.
    Combines features from data-driven env and planning-focused env.
    """

    EPS = 0.05  # interpolation step for edge checking and visualization

    def __init__(self, GUI=False, kuka_file="kuka_iiwa/model.urdf"):
        self.GUI = GUI
        self.kuka_file = kuka_file
        self.dim = 3
        self.obstacles = []
        self.start_state = None
        self.goal_state = None

        # Connect to PyBullet
        if GUI:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=[target[0], target[1], 0.7])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load plane and robot
        self.planeId = p.loadURDF("plane.urdf")
        self.kukaId = p.loadURDF(self.kuka_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

        # Configuration space
        self.config_dim = p.getNumJoints(self.kukaId)
        self.pose_range = [(p.getJointInfo(self.kukaId, j)[8], p.getJointInfo(self.kukaId, j)[9])
                           for j in range(self.config_dim)]
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.kukaEndEffectorIndex = self.config_dim - 1

        p.stepSimulation()

    @property
    def init_state(self):
        return self.start_state

    @property
    def goal_state(self):
        return self.goal_state

    @property
    def bound(self):
        return np.array(self.pose_range).T.reshape(-1)

    def _state_fp(self, state):
        return self.is_state_free(state)

    def _edge_fp(self, state1, state2):
        return self.is_edge_free(state1, state2)

    # ----------------- Obstacles -----------------
    def add_box_obstacle(self, half_extents, base_position):
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=np.random.uniform(0,1,3).tolist() + [0.8])
        body_id = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col_id,
                                    baseVisualShapeIndex=vis_id,
                                    basePosition=base_position)
        self.obstacles.append((half_extents, base_position))
        return body_id

    def reset_obstacles(self):
        p.resetSimulation()
        self.obstacles = []
        self.planeId = p.loadURDF("plane.urdf")
        self.kukaId = p.loadURDF(self.kuka_file, [0,0,0], [0,0,0,1], useFixedBase=True)

    # ----------------- Configuration -----------------
    def set_config(self, state, kukaId=None):
        if kukaId is None:
            kukaId = self.kukaId
        for j in range(self.config_dim):
            p.resetJointState(kukaId, j, state[j])
        p.performCollisionDetection()

    # ----------------- Collision Checking -----------------
    def valid_state(self, state):
        return (state >= np.array(self.pose_range)[:,0]).all() and (state <= np.array(self.pose_range)[:,1]).all()

    def is_state_free(self, state):
        if not self.valid_state(state):
            return False
        self.set_config(state)
        return len(p.getContactPoints(self.kukaId)) == 0

    def is_edge_free(self, state, new_state):
        assert state.size == new_state.size
        if not self.valid_state(state) or not self.valid_state(new_state):
            return False
        if not self.is_state_free(state) or not self.is_state_free(new_state):
            return False

        disp = new_state - state
        d = self.distance(state, new_state)
        K = max(int(d / self.EPS), 1)
        for k in range(K + 1):
            interp_state = state + k / K * disp
            if not self.is_state_free(interp_state):
                return False
        return True

    # ----------------- Sampling -----------------
    def uniform_sample(self, n=1):
        sample = np.random.uniform(np.array(self.pose_range)[:,0], np.array(self.pose_range)[:,1], size=(n, self.config_dim))
        if n == 1:
            return sample.reshape(-1)
        return sample

    def sample_start_goal(self, max_attempts=100):
        for _ in range(max_attempts):
            start = self.uniform_sample()
            goal = self.uniform_sample()
            if self.is_state_free(start) and self.is_state_free(goal) and np.linalg.norm(start - goal) > 0.1:
                self.start_state = start
                self.goal_state = goal
                return start, goal
        return None, None

    # ----------------- Planning Utilities -----------------
    def distance(self, from_state, to_state):
        diff = to_state - from_state
        return np.sqrt(np.sum(diff**2))

    def interpolate(self, from_state, to_state, ratio):
        interp_state = from_state + ratio * (to_state - from_state)
        interp_state = np.clip(interp_state, np.array(self.pose_range)[:,0], np.array(self.pose_range)[:,1])
        return interp_state

    def in_goal_region(self, state, eps=None):
        if eps is None:
            eps = self.EPS
        return self.distance(state, self.goal_state) < eps and self.is_state_free(state)
    
    # ----------------- Visualization -----------------
    def plot_path(self, path, make_gif=False):
        path = np.array(path)
        self.set_config(path[0])
        goal_kuka = p.loadURDF(self.kuka_file, [0,0,0], [0,0,0,1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        self.set_config(path[-1], goal_kuka)
        gifs = []
        for idx in range(len(path)-1):
            disp = path[idx+1] - path[idx]
            d = self.distance(path[idx], path[idx+1])
            K = max(int(d/self.EPS),1)
            new_kuka = p.loadURDF(self.kuka_file, [0,0,0], [0,0,0,1], useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
            for data in p.getVisualShapeData(new_kuka):
                color = list(data[-1])
                color[-1] = 0.5
                p.changeVisualShape(new_kuka, data[1], rgbaColor=color)
            for k in range(K+1):
                c = path[idx] + k/K*disp
                self.set_config(c, new_kuka)
                p.performCollisionDetection()
                if make_gif:
                    image = p.getCameraImage(width=1080, height=900, lightDirection=[1,1,1])[2]
                    gifs.append(image)
        return gifs
