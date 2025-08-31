import json
import numpy as np
from os import makedirs
from os.path import join
from environment.kuka_env import KukaEnv

# ---------------- Configuration ----------------
config_name = "kuka_random_3d"
dataset_dir = join("data", config_name)

config = {
    'random_seed': 42,
    'num_obstacles_range': [5, 15],       # 障碍物数量范围
    'box_size_range': [0.05, 0.2],        # 障碍物尺寸范围
    'space_range': [-1, 1],               # workspace 边界 (障碍物放置空间)
    'num_samples_per_env': 5,             # 每个环境采样的 start-goal 数量
    'redundant_env_size_scale': 2,        # 多生成一些以剔除无效
    'train_env_size': 20,
    'val_env_size': 5,
    'test_env_size': 5,
}

np.random.seed(config['random_seed'])
env_size = {
    'train': config['train_env_size'],
    'val': config['val_env_size'],
    'test': config['test_env_size'],
}

# ---------------- Helper Functions ----------------
def generate_random_obstacles(env):
    """在 workspace 中生成随机立方体障碍物"""
    env.obstacles = []
    num_obstacles = np.random.randint(config['num_obstacles_range'][0], config['num_obstacles_range'][1])
    for _ in range(num_obstacles):
        half_extents = np.random.uniform(config['box_size_range'][0], config['box_size_range'][1], 3)
        base_position = np.random.uniform(config['space_range'][0], config['space_range'][1], 3)
        env.add_box_obstacle(half_extents, base_position)

def sample_collision_free_state(env, max_attempts=100):
    """在关节空间中采样一个无碰撞的状态"""
    for _ in range(max_attempts):
        state = env.uniform_sample()
        if env.is_state_free(state):
            return state
    return None

def generate_start_goal(env, max_attempts=100):
    """采样一对可行的 start-goal 状态"""
    for _ in range(max_attempts):
        start = sample_collision_free_state(env)
        goal = sample_collision_free_state(env)
        if start is not None and goal is not None and np.linalg.norm(start - goal) > 0.1:
            return start, goal
    return None, None

# ---------------- Dataset Generation ----------------
total_env_count = 0
invalid_env_count = 0

for mode in ['train', 'val', 'test']:
    mode_dir = join(dataset_dir, mode)
    makedirs(mode_dir, exist_ok=True)
    mode_env_list = []

    while len(mode_env_list) < env_size[mode] * config['redundant_env_size_scale']:
        total_env_count += 1
        env = KukaEnv(GUI=False)
        generate_random_obstacles(env)

        valid_env = True
        x_start_list, x_goal_list = [], []

        for _ in range(config['num_samples_per_env']):
            x_start, x_goal = generate_start_goal(env)
            if x_start is None:
                valid_env = False
                break
            x_start_list.append(x_start.tolist())
            x_goal_list.append(x_goal.tolist())

        if not valid_env:
            invalid_env_count += 1
            print(f"Invalid env: {invalid_env_count}/{total_env_count}")
            continue

        # 转换障碍物格式 [x, y, z, w, h, d]
        box_obstacles = []
        for obs in env.obstacles:
            half_extents, position = obs
            x, y, z = position
            w, h, d = half_extents * 2
            box_obstacles.append([x, y, z, w, h, d])

        env_dict = {
            # workspace 边界 (对齐原始 raw 脚本字段名)
            'env_dims': [
                config['space_range'][0], config['space_range'][0], config['space_range'][0],
                config['space_range'][1], config['space_range'][1], config['space_range'][1]
            ],
            # 额外保留 KUKA 的关节空间边界
            'joint_bounds': env.bound.tolist(),
            'box_obstacles': box_obstacles,
            'ball_obstacles': [],  # 保持和原始脚本一致，即使为空
            'start': x_start_list,
            'goal': x_goal_list
        }
        mode_env_list.append(env_dict)

        # 保存 JSON（覆盖写）
        with open(join(mode_dir, "raw_envs.json"), "w") as f:
            json.dump(mode_env_list, f, indent=2)

        if len(mode_env_list) % 5 == 0:
            print(f"{len(mode_env_list)} {mode} envs and {len(mode_env_list)*config['num_samples_per_env']} samples saved.")

print(f"Finished generating dataset. Total environments: {total_env_count}, Invalid: {invalid_env_count}")
