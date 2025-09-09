import json
import traceback, os, sys
from os import makedirs
from os.path import join
from matplotlib import pyplot as plt
import numpy as np
import pybullet as p
import time
from tqdm import tqdm
from multiprocessing import Pool
from environment.kuka_env import KukaEnv
from path_planning_classes_3d.bit_star import BITStar
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Configuration ----------------
config_name = "kuka_random_3d"
dataset_dir = join("data", config_name)

config = {
    'random_seed': 42,
    'num_obstacles_range': [5, 15],
    'box_size_range': [0.05, 0.2],
    'space_range': [-1, 1],
    'num_samples_per_env': 5,
    'redundant_env_size_scale': 2,
    'train_env_size': 4000,
    'val_env_size': 500,
    'test_env_size': 500,
    'GUI': False,
    'refine_time_budget': 10,  
    'time_budget': 30, 
}

np.random.seed(config['random_seed'])

# ---------------- Helper Functions ----------------
def generate_bitstar_path(env, x_start, x_goal, max_attempts=3, gui=False, refine_time_budget=np.inf,time_budget=np.inf):
    env.init_state = np.array(x_start)
    env.goal_state = np.array(x_goal)
    for attempt in range(max_attempts):
        bitstar = BITStar(env, maxIter=5, batch_size=200,T=1000000)
        samples, edges, collision_count, goal_score, total_samples, exec_time = bitstar.plan(pathLengthLimit=np.inf, refine_time_budget=refine_time_budget,time_budget=time_budget)
        path = bitstar.get_best_path()
        if path and len(path) > 0:
            if gui:
                visualize_path(env, path)
            return path, exec_time
    return None, None

def visualize_path(env, path, step_ratio=0.1, snapshot_interval=50, ee_marker_scale=0.05):
    if not env.GUI:
        raise RuntimeError("GUI must be enabled to visualize path")
    path = np.array(path)
    interp_counter = 0
    red_spheres, kuka_snaps = [], []

    try:
        env.set_config(path[0], env.kukaId)
        goal_kuka = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1],
                                useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
        env.set_config(path[-1], goal_kuka)
        for i in range(len(path) - 1):
            q_start, q_end = path[i], path[i + 1]
            disp = q_end - q_start
            n_steps = max(int(np.linalg.norm(disp) / step_ratio), 1)

            for k in range(n_steps + 1):
                q_interp = q_start + (k / n_steps) * disp
                # 克隆透明机械臂
                if interp_counter % snapshot_interval == 0:
                    kuka_snap = p.loadURDF(env.kuka_file, [0, 0, 0], [0, 0, 0, 1],
                                           useFixedBase=True, flags=p.URDF_IGNORE_COLLISION_SHAPES)
                    for data in p.getVisualShapeData(kuka_snap):
                        color = list(data[-1])
                        color[-1] = 0.3
                        p.changeVisualShape(kuka_snap, data[1], rgbaColor=color)
                    kuka_snaps.append(kuka_snap)
                env.set_config(q_interp, kuka_snaps[-1] if kuka_snaps else env.kukaId)
                ee_pos = p.getLinkState(kuka_snaps[-1] if kuka_snaps else env.kukaId, env.kukaEndEffectorIndex)[0]
                red_ball = p.loadURDF("sphere2red.urdf", ee_pos, globalScaling=ee_marker_scale,
                                      flags=p.URDF_IGNORE_COLLISION_SHAPES)
                red_spheres.append(red_ball)
                interp_counter += 1
                p.stepSimulation()
        # print("路径可视化完成，按回车关闭并清空快照与红球...")
        # input()
    finally:
        for ball in red_spheres: p.removeBody(ball)
        for snap in kuka_snaps: p.removeBody(snap)
        p.removeBody(goal_kuka)
        p.removeAllUserDebugItems()


def visualize_env_sample(mode, env_id, sample_idx, dataset_dir="data/kuka_random_3d"):
    """
    可视化指定环境和 start-goal 样本的运动路径，并绘制关节角度曲线
    :param mode: 'train' / 'val' / 'test'
    :param env_id: 环境编号
    :param sample_idx: 样本编号 (即 start-goal 对序号)
    :param dataset_dir: 数据集目录
    """
    mode_dir = join(dataset_dir, mode)

    # 1. 读取环境配置
    with open(join(mode_dir, "envs.json"), "r") as f:
        envs = json.load(f)
    env_dict = next(e for e in envs if e["env_id"] == env_id)
    if env_dict is None:
        raise ValueError(f"Env id {env_id} not found in {mode} set")
    # 2. 读取路径文件
    path_file = join(mode_dir, "bitstar_paths", f"{env_id}_{sample_idx}.txt")
    if not os.path.exists(path_file):
        raise ValueError(f"Path file {path_file} not found")
    path = np.loadtxt(path_file, delimiter=',')

    # 3. 创建环境并添加障碍物
    env = KukaEnv(GUI=True)
    env.reset_obstacles()
    env.reset_arm()
    for obs in env_dict["box_obstacles"]:
        half_extents = np.array(obs[3:]) / 2.0
        base_position = np.array(obs[:3])
        env.add_box_obstacle(half_extents, base_position)

    # 4. 可视化路径
    visualize_path(env, path)

    # 5. 绘制关节角度变化曲线
    path = np.array(path)  # (N, D)
    num_points, num_joints = path.shape
    plt.figure(figsize=(10, 6))
    for j in range(num_joints):
        plt.plot(range(num_points), path[:, j], label=f"Joint {j}", marker='o', markersize=5)
    plt.xlabel("Path step")
    plt.ylabel("Joint angle (rad)")
    plt.title(f"Env {env_id}, Sample {sample_idx} Joint Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()

    env.close()


def save_path_and_env(mode_path_dir, env_idx, path_list, env_dict, mode_dir, mode_env_list):
    for path_idx, path in enumerate(path_list):
        path_np = np.array(path)
        np.savetxt(join(mode_path_dir, f"{env_idx}_{path_idx}.txt"), path_np, fmt='%.6f', delimiter=',')
    with open(join(mode_dir, "envs.json"), "w") as f:
        json.dump(mode_env_list, f)

def generate_bitstar_dataset(mode, env_size, redundant_mode_env_list, config, dataset_dir):
    print(f"Generating BIT* paths for {mode} set...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "bitstar_paths")
    makedirs(mode_path_dir, exist_ok=True)
    mode_env_list = []
    total_env_count = 0
    invalid_env_count = 0

    env = KukaEnv(GUI=config['GUI'])
    with tqdm(total=env_size, desc=f"[{mode}]", position=0, leave=True) as pbar:
        while len(mode_env_list) < env_size:
            if total_env_count >= len(redundant_mode_env_list):
                print("redundant_mode_env not enough")
                break
            total_env_count += 1
            env.reset_obstacles()
            env.reset_arm()
            env_dict = redundant_mode_env_list[total_env_count - 1]
            print(f"env_idx={total_env_count - 1}")
            for obs in env_dict['box_obstacles']:
                half_extents = np.array(obs[3:]) / 2.0
                base_position = np.array(obs[:3])
                env.add_box_obstacle(half_extents, base_position)
            x_start_list, x_goal_list = env_dict['start'], env_dict['goal']

            valid_env = True
            path_list, exec_time_list = [], []

            for idx in range(config['num_samples_per_env']):
                x_start, x_goal = x_start_list[idx], x_goal_list[idx]
                path, exec_time = generate_bitstar_path(env, x_start, x_goal, gui=config['GUI'], refine_time_budget=config['refine_time_budget'], time_budget=config['time_budget'])
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
            pbar.update(1)
            save_path_and_env(mode_path_dir, env_idx, path_list, env_dict, mode_dir, mode_env_list)

    print(f"Finished {mode} set. Total valid envs: {len(mode_env_list)}, Invalid: {invalid_env_count}")


def process_single_env(args):
    env_idx, env_dict, num_samples, gui, refine_time_budget, time_budget, mode_path_dir = args
    env_dict['env_id'] = env_idx   # 加入固定 id
    pid = os.getpid()
    # ====== 跳过已生成的环境 ======
    all_exist = True
    for path_idx in range(num_samples):
        filename = join(mode_path_dir, f"{env_idx}_{path_idx}.txt")
        if not os.path.exists(filename):
            all_exist = False
            break
    if all_exist:
        print(f"[Subprocess {pid}] skip env {env_idx}, already done", flush=True)
        return None
    # =============================
    try:
        print(f"[Subprocess {pid}] start env {env_idx}", flush=True)
        env = KukaEnv(GUI=gui)

        # 添加障碍物
        for obs in env_dict['box_obstacles']:
            half_extents = np.array(obs[3:]) / 2.0
            base_position = np.array(obs[:3])
            env.add_box_obstacle(half_extents, base_position)

        x_start_list, x_goal_list = env_dict['start'], env_dict['goal']
        path_list, exec_time_list = [], []

        for i in range(num_samples):
            x_start, x_goal = x_start_list[i], x_goal_list[i]
            path, exec_time = generate_bitstar_path(env, x_start, x_goal,
                                                   gui=gui,
                                                   refine_time_budget=refine_time_budget,
                                                   time_budget=time_budget)
            if path is None:
                print(f"[Subprocess {pid}] env {env_idx} failed sample {i}", flush=True)
                return None
            path_list.append(path)
            exec_time_list.append(exec_time)

        env_dict['bitstar_time'] = exec_time_list
        print(f"[Subprocess {pid}] finished env {env_idx}", flush=True)
        return env_dict, path_list

    except Exception as e:
        # 把异常打印出来（子进程的 stderr 会回到父进程控制台）
        tb = traceback.format_exc()
        print(f"[Subprocess {pid}] Exception processing env {env_idx}:\n{tb}", flush=True)
        return None

    finally:
        # 强制清理 env（如果 env 存在且有 close）
        try:
            if 'env' in locals():
                if hasattr(env, 'close'):
                    env.close()
                else:
                    # 最少调用 pybullet disconnect
                    try:
                        p.disconnect()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Subprocess {pid}] cleanup exception: {e}", flush=True)

def save_env_paths(mode_path_dir, mode_env_list):
    for env_item in mode_env_list:
        env_id = env_item['env_id']
        if len(mode_path_dir)>1000:
            path_list = env_item.pop('paths', [])
        for path_idx, path in enumerate(path_list):
            path_np = np.array(path)
            filename = join(mode_path_dir, f"{env_id}_{path_idx}.txt")
            np.savetxt(filename, path_np, fmt='%.6f', delimiter=',')
            # print(f"Saved path: {filename}")
    return mode_env_list

def generate_bitstar_dataset_parallel(mode, env_size, redundant_mode_env_list, config, dataset_dir, save_interval=10, n_process=4):
    print(f"Generating BIT* paths for {mode} set (parallel, {n_process} processes)...")
    mode_dir = join(dataset_dir, mode)
    mode_path_dir = join(mode_dir, "bitstar_paths")
    makedirs(mode_path_dir, exist_ok=True)
    mode_env_list = []
    final_file = join(mode_dir, "envs.json")

    done_envs = set()
    if os.path.exists(final_file):
        with open(final_file, "r", encoding="utf-8") as f_json:
            try:
                old_envs = json.load(f_json)
                for e in old_envs:
                    done_envs.add(e.get("env_id", -1))
                if len(done_envs)>=env_size:
                    print(f"{mode} set already completed with {len(done_envs)} envs.")
                    return
                mode_env_list.extend(old_envs)
                print(f"已有 {len(done_envs)} 个环境完成，继续剩余部分...")
            except Exception:
                pass
            
    args_list = []
    for i, env_dict in enumerate(redundant_mode_env_list):
        if i in done_envs:
            continue
        args_list.append((i, env_dict, config['num_samples_per_env'], False,
                          config['refine_time_budget'], config['time_budget'], mode_path_dir))

    with ProcessPoolExecutor(max_workers=n_process) as executor:
        futures = {executor.submit(process_single_env, args): args[0] for args in args_list}

        with tqdm(
                total=env_size,
                initial=len(done_envs),   # 让进度条从已完成的环境数量开始
                desc=f"[{mode}] 有效环境进度"
            ) as pbar:
            for f in as_completed(futures):
                result = f.result()
                if result is not None:
                    env_dict, path_list = result
                    env_dict['paths'] = path_list
                    mode_env_list.append(env_dict)

                    # 立刻保存路径（env_id 固定）
                    save_env_paths(mode_path_dir, [env_dict])
                    pbar.update(1)

                    # 阶段性保存 JSON
                    if len(mode_env_list) % save_interval == 0:
                        with open(final_file, "w", encoding="utf-8") as f_json:
                            json.dump(mode_env_list, f_json, ensure_ascii=False)
                        print(f"保存 {final_file}, 当前已有 {len(mode_env_list)} 个环境")

                # 终止条件：够了 env_size 个有效环境
                if len(mode_env_list) >= env_size:
                    print("达到所需环境数量，提前终止")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # 保存最终结果
    print("Saving final paths and envs...")
    save_env_paths(mode_path_dir, mode_env_list)
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(mode_env_list, f, ensure_ascii=False)

    print(f"Finished {mode} set. Total valid envs: {len(mode_env_list)}")


# ---------------- Main Program ----------------
if __name__ == "__main__":
    env_size_dict = {
        'train': config['train_env_size'],
        'val': config['val_env_size'],
        'test': config['test_env_size'],
    }
    for mode in ['train', 'val', 'test']:
        mode_dir = join(dataset_dir, mode)
        with open(join(mode_dir, "raw_envs.json"), "r") as f:
            redundant_mode_env_list = json.load(f)
            print(f"Loaded {len(redundant_mode_env_list)} redundant envs for {mode} set")
        # generate_bitstar_dataset(mode, env_size_dict[mode], redundant_mode_env_list, config, dataset_dir)
        # generate_bitstar_dataset_parallel(mode, env_size_dict[mode], redundant_mode_env_list, config, dataset_dir, n_process=6)
    # visualize_env_sample('train', env_id=300, sample_idx=2, dataset_dir=dataset_dir)
