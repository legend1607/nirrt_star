import json
import time
from os.path import join
import numpy as np
import yaml

from environment.kuka_env import KukaEnv
from datasets.point_cloud_mask_utils import get_point_cloud_mask_around_points
from datasets_3d.point_cloud_mask_utils_3d import generate_configuration_space_point_cloud_kuka

def save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False):
    raw_dataset_saved = {}
    for k in raw_dataset.keys():
        if k == 'token':
            raw_dataset_saved[k] = np.array(raw_dataset[k])
        else:
            raw_dataset_saved[k] = np.stack(raw_dataset[k], axis=0)  # (b, n_points, ...)
    filename = mode+"_tmp.npz" if tmp else mode+".npz"
    np.savez(join(dataset_dir, filename), **raw_dataset_saved)

config_name = "kuka_random_3d"
with open(join("env_configs", config_name+".yml"), 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

dataset_dir = join("data", config_name)

for mode in ['train', 'val', 'test']:
    with open(join(dataset_dir, mode, "envs.json"), 'r') as f:
        env_list = json.load(f)

    raw_dataset = {
        'token': [],
        'pc': [],
        'start': [],
        'goal': [],
        'free': [],
        'bitstar': []
    }

    start_time = time.time()
    for env_idx, env_dict in enumerate(env_list):
        env = KukaEnv(GUI=False)
        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3]) + half_extents
            env.add_box_obstacle(half_extents, base_position)

        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            env.start_state = np.array(x_start)
            env.goal_state = np.array(x_goal)
            sample_title = f"{env_idx}_{sample_idx}"

            # 读取 BIT* 路径
            path = np.loadtxt(
                join(dataset_dir, mode, "bitstar_paths", sample_title+".txt"),
                delimiter=','
            )  # shape: (n_points, config_dim)

            # 生成点云
            pc = generate_configuration_space_point_cloud_kuka(
                env,
                config['n_points'],
                over_sample_scale=config.get('over_sample_scale', 1.0)
            )

            # 生成 mask
            around_start_mask = get_point_cloud_mask_around_points(
                pc,
                np.array(x_start)[np.newaxis, :],
                neighbor_radius=config['start_radius']
            )

            around_goal_mask = get_point_cloud_mask_around_points(
                pc,
                np.array(x_goal)[np.newaxis, :],
                neighbor_radius=config['goal_radius']
            )

            around_path_mask = get_point_cloud_mask_around_points(
                pc,
                path,
                neighbor_radius=config['path_radius']
            )

            freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

            # 存入 dataset
            raw_dataset['token'].append(f"{mode}-{sample_title}")
            raw_dataset['pc'].append(pc.astype(np.float32))
            raw_dataset['start'].append(around_start_mask.astype(np.float32))
            raw_dataset['goal'].append(around_goal_mask.astype(np.float32))
            raw_dataset['free'].append(freespace_mask.astype(np.float32))
            raw_dataset['bitstar'].append(around_path_mask.astype(np.float32))

        if (env_idx+1) % 5 == 0:
            time_left = (time.time() - start_time) * (len(env_list)/(env_idx+1) - 1)/60
            print(f"{mode} {env_idx+1}/{len(env_list)}, remaining time: {int(time_left)} min")

    save_raw_dataset(raw_dataset, dataset_dir, mode, tmp=False)
