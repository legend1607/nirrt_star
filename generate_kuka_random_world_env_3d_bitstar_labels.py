import json
import time
from os import makedirs
from os.path import join

import yaml
import numpy as np

from environment.kuka_env import KukaEnv
from path_planning_classes_3d.bit_star import BITStar

# ---------------- Helper Functions ----------------
def generate_bitstar_path(env, x_start, x_goal, max_attempts=3):
    env.start_state = np.array(x_start)
    env.goal_state = np.array(x_goal)
    for attempt in range(max_attempts):
        bitstar = BITStar(env, maxIter=5, batch_size=200)
        samples, edges, collision_count, goal_score, total_samples, exec_time = bitstar.plan(np.inf)
        path = bitstar.get_best_path()
        if path and len(path) > 0:
            return path, exec_time
    return None, None

# ---------------- Load Configuration ----------------
config_name = "kuka_random_3d"
with open(join("env_configs", config_name+".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_dir = join("data", config_name)
np.random.seed(config['random_seed'])

num_samples_per_env = config['num_samples_per_env']

# ---------------- Main Loop ----------------
for mode in ['train', 'val', 'test']:
    print(f"Generating BIT* paths for {mode} set...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "bitstar_paths")
    makedirs(mode_path_dir, exist_ok=True)

    # 读取 raw 环境及 start/goal
    with open(join(mode_dir, "raw_envs.json"), "r") as f:
        redundant_mode_env_list = json.load(f)

    mode_env_list = []
    total_env_count = 0
    invalid_env_count = 0

    while len(mode_env_list) < config[mode+'_env_size']:
        total_env_count += 1
        env_dict = redundant_mode_env_list[total_env_count - 1]

        env = KukaEnv(GUI=False)
        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])  # 注意这里不加 half_extents
            env.add_box_obstacle(half_extents, base_position)

        x_start_list, x_goal_list = env_dict['start'], env_dict['goal']

        valid_env = True
        path_list, exec_time_list = [], []

        for start_goal_idx in range(num_samples_per_env):
            x_start, x_goal = x_start_list[start_goal_idx], x_goal_list[start_goal_idx]
            path, exec_time = generate_bitstar_path(env, x_start, x_goal)
            if path is None:
                valid_env = False
                break
            path_list.append(path)
            exec_time_list.append(exec_time)

        if not valid_env:
            invalid_env_count += 1
            print(f"Invalid env: {invalid_env_count}/{total_env_count}")
            continue

        env_dict['bitstar_time'] = exec_time_list
        mode_env_list.append(env_dict)
        env_idx = len(mode_env_list) - 1

        # 保存路径文件
        for path_idx, path in enumerate(path_list):
            path_np = np.array(path)
            np.savetxt(join(mode_path_dir, f"{env_idx}_{path_idx}.txt"), path_np, fmt='%.6f', delimiter=',')

        # 每生成 5 个环境打印一次状态
        if len(mode_env_list) % 5 == 0:
            print(f"{len(mode_env_list)} {mode} envs and {num_samples_per_env*len(mode_env_list)} samples processed.")

    # ---------------- 写入 JSON ----------------
    with open(join(mode_dir, "envs.json"), "w") as f:
        json.dump(mode_env_list, f, indent=2)

    print(f"Finished {mode} set. Total valid envs: {len(mode_env_list)}, Invalid: {invalid_env_count}")
