import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import os
import sys
import time
import traceback
import numpy as np
import pybullet as p
from os.path import join, exists
from os import makedirs

from tqdm import tqdm

from environment.kuka_env import KukaEnv
from datasets.point_cloud_mask_utils import (generate_rectangle_point_cloud,
    get_point_cloud_mask_around_points)
from datasets_3d.point_cloud_mask_utils_3d import generate_rectangle_point_cloud_3d

# ---------------- 配置参数 ----------------
config = {
    'dataset_dir': "data/kuka_random_3d",
    'modes': ['train'],
    'n_points': 10000,
    'over_sample_scale': 10,
    'point_cloud_clearance': 0.0,  # 生成点云时避开障碍物距离
    'start_radius': 0.2,           # 起点周围点云半径
    'goal_radius': 0.2,            # 终点周围点云半径
    'path_radius': 0.1,            # 路径周围点云半径
    'save_interval': 20, 
}

def generate_joint_space_point_cloud(env, n_points, over_sample_scale=5):
    """
    采样关节空间点云，保证碰撞自由
    - env: KukaEnv
    - n_points: 最终采样点数
    - over_sample_scale: 超采样倍数，用于拒绝采样
    """
    max_attempts = n_points * over_sample_scale
    point_cloud = []
    attempts = 0
    while len(point_cloud) < n_points and attempts < max_attempts:
        q = env.uniform_sample()
        if env.is_state_free(q):
            point_cloud.append(q)
        attempts += 1

    if len(point_cloud) < n_points:
        print(f"Warning: only sampled {len(point_cloud)} points out of requested {n_points}")

    point_cloud = np.array(point_cloud)

    # 如果多采样了，可以随机下采样到 n_points
    if len(point_cloud) > n_points:
        idx = np.random.choice(len(point_cloud), n_points, replace=False)
        point_cloud = point_cloud[idx]

    return point_cloud

# ---------------- 保存函数 ----------------
def save_dataset(pc_dataset, dataset_dir, mode):
    """保存 dataset 到固定文件名 mode.npz"""
    makedirs(dataset_dir, exist_ok=True)
    filename = join(dataset_dir, f"{mode}.npz")
    pc_dataset_saved = {k: (np.array(v) if k == 'token' else np.stack(v, axis=0))
                         for k, v in pc_dataset.items()}
    np.savez(filename, **pc_dataset_saved)
    print(f"{mode} dataset saved: {len(pc_dataset['token'])} samples to {filename}")

# ---------------- 生成点云和 mask ----------------
def generate_point_cloud(mode, env_list,config):
    pc_dataset = {
        'token': [],
        'pc': [],
        'start': [],
        'goal': [],
        'free': [],
        'bitstar': [],
    }

    env = KukaEnv(GUI=False)
    start_time = time.time()
    pbar = tqdm(total=len(env_list), desc=f"{mode} dataset progress")
    for env_dict in env_list:
        env_idx=env_dict['env_id']
        print(f"Processing {mode} env {env_idx}...")
        env.reset_obstacles()
        env.reset_arm()

        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            env.init_state = np.array(x_start)
            env.goal_state = np.array(x_goal)

            sample_title = f"{env_idx}_{sample_idx}"
            token = f"{mode}-{sample_title}"

            # 点云生成
            pc = generate_joint_space_point_cloud(env, n_points=config['n_points'], over_sample_scale=config['over_sample_scale'])

            # BIT* 路径
            if "paths" in env_dict:
                path = np.array(env_dict['paths'][sample_idx])
            else:
                path_file = join(config['dataset_dir'], mode, "bitstar_paths", f"{env_idx}_{sample_idx}.txt")
                path = np.loadtxt(path_file, delimiter=',')

            # mask 生成
            around_start_mask = get_point_cloud_mask_around_points(
                pc, np.array(x_start)[np.newaxis, :], neighbor_radius=config['start_radius']
            )
            around_goal_mask = get_point_cloud_mask_around_points(
                pc, np.array(x_goal)[np.newaxis, :], neighbor_radius=config['goal_radius']
            )
            around_path_mask = get_point_cloud_mask_around_points(
                pc, path, neighbor_radius=config['path_radius']
            )
            freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

            # 保存到 dataset
            pc_dataset['token'].append(token)
            pc_dataset['pc'].append(pc.astype(np.float32))
            pc_dataset['start'].append(around_start_mask.astype(np.float32))
            pc_dataset['goal'].append(around_goal_mask.astype(np.float32))
            pc_dataset['free'].append(freespace_mask.astype(np.float32))
            pc_dataset['bitstar'].append(around_path_mask.astype(np.float32))
        pbar.update(1)

        if len(pc_dataset['token']) % config['save_interval'] == 0:
            save_dataset(pc_dataset, config['dataset_dir'], mode)
            
    save_dataset(pc_dataset, config['dataset_dir'], mode)
    env.close()


def process_env(args):
    """子进程函数：处理单个 env_dict"""
    mode, env_dict, config = args
    env_idx = env_dict['env_id']
    pid = os.getpid() 
    env = None
    print(f"[Subprocess {pid}] processing env {env_idx}", flush=True)
    try:
        env = KukaEnv(GUI=False)
        env.reset_obstacles()
        env.reset_arm()

        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        tokens, pcs, starts, goals, frees, bitstars = [], [], [], [], [], []

        for sample_idx, (x_start, x_goal) in enumerate(zip(env_dict['start'], env_dict['goal'])):
            env.init_state = np.array(x_start)
            env.goal_state = np.array(x_goal)

            sample_title = f"{env_idx}_{sample_idx}"
            token = f"{mode}-{sample_title}"

            # 点云生成
            pc = generate_joint_space_point_cloud(
                env, n_points=config['n_points'],
                over_sample_scale=config['over_sample_scale']
            )

            # BIT* 路径
            if "paths" in env_dict:
                path = np.array(env_dict['paths'][sample_idx])
            else:
                path_file = join(config['dataset_dir'], mode, "bitstar_paths", f"{env_idx}_{sample_idx}.txt")
                path = np.loadtxt(path_file, delimiter=',')

            # mask 生成
            around_start_mask = get_point_cloud_mask_around_points(
                pc, np.array(x_start)[np.newaxis, :], neighbor_radius=config['start_radius']
            )
            around_goal_mask = get_point_cloud_mask_around_points(
                pc, np.array(x_goal)[np.newaxis, :], neighbor_radius=config['goal_radius']
            )
            around_path_mask = get_point_cloud_mask_around_points(
                pc, path, neighbor_radius=config['path_radius']
            )
            freespace_mask = (1 - around_start_mask) * (1 - around_goal_mask)

            # 收集结果
            tokens.append(token)
            pcs.append(pc.astype(np.float32))
            starts.append(around_start_mask.astype(np.float32))
            goals.append(around_goal_mask.astype(np.float32))
            frees.append(freespace_mask.astype(np.float32))
            bitstars.append(around_path_mask.astype(np.float32))
            
        print(f"[Subprocess {pid}] finished env {env_idx}", flush=True)
        return {
            'token': tokens,
            'pc': pcs,
            'start': starts,
            'goal': goals,
            'free': frees,
            'bitstar': bitstars,
        }

    except Exception as e:
        # 把异常和 traceback 打印出来
        tb = traceback.format_exc()
        print(f"[Subprocess {pid}] Exception processing env {env_idx}:\n{tb}", flush=True)
        return None

    finally:
        # 强制清理 env
        try:
            if env is not None:
                if hasattr(env, 'close'):
                    env.close()
                else:
                    try:
                        p.disconnect()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Subprocess {pid}] cleanup exception: {e}", flush=True)


def generate_point_cloud_parallel(mode, env_list, config, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    dataset_dir = join(config['dataset_dir'], mode)
    final_file = join(dataset_dir, f"{mode}.npz")

    # ---- 断点续跑：尝试加载已有数据 ----
    pc_dataset = {k: [] for k in ['token', 'pc', 'start', 'goal', 'free', 'bitstar']}
    finished_tokens = set()
    if exists(final_file):
        print(f"[Resume] Loading existing dataset {final_file}")
        loaded = np.load(final_file, allow_pickle=True)
        for k in pc_dataset.keys():
            pc_dataset[k] = list(loaded[k])
        finished_tokens = set(pc_dataset['token'])
        print(f"[Resume] 已完成 {len(finished_tokens)} samples, 将跳过重复 env")

    # ---- 筛选未完成的 env ----
    env_list_to_run = []
    for env in env_list:
        env_id = env['env_id']
        token_prefix = f"{mode}-{env_id}_"
        if any(t.startswith(token_prefix) for t in finished_tokens):
            continue
        env_list_to_run.append(env)

    print(f"[Main] {mode}: {len(env_list)} total envs, {len(env_list_to_run)} remaining after resume")

    if not env_list_to_run:
        print(f"[Main] {mode} 已完成，无需继续")
        return

    # ---- 并行运行 ----
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_env, (mode, e, config)): e['env_id'] for e in env_list_to_run}

        with tqdm(total=len(env_list_to_run), desc=f"[{mode}] env progress") as pbar:
            for i, f in enumerate(as_completed(futures)):
                env_id = futures[f]
                result = f.result()
                if result is not None:
                    for k in pc_dataset.keys():
                        pc_dataset[k].extend(result[k])
                    pbar.update(1)

                    if (i + 1) % config['save_interval'] == 0:
                        save_dataset(pc_dataset, dataset_dir, mode)
                else:
                    print(f"[Error] Subprocess failed on env {env_id}, exiting...", flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)

    save_dataset(pc_dataset, dataset_dir, mode)
    print(f"[Main] {mode} 完成，结果保存到 {final_file}")

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    for mode in config['modes']:
        envs_file = join(config['dataset_dir'], mode, "envs.json")
        if not exists(envs_file):
            print(f"环境文件不存在: {envs_file}")
            continue

        with open(envs_file, 'r') as f:
            env_list = json.load(f)
        if all("env_id" in e for e in env_list):
            env_list = sorted(env_list, key=lambda e: e["env_id"])

        generate_point_cloud(mode,env_list,config)
        # generate_point_cloud_parallel(mode, env_list, config, num_workers=8)