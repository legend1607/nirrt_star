import time
import pickle
import argparse
from tqdm import tqdm
from copy import copy
from os import makedirs
from os.path import join, exists
from importlib import import_module
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='rrt_star', 
        help='rrt_star, irrt_star, nrrt_star, nirrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs')
    parser.add_argument('--device', default=None, help='cuda or cpu')
    parser.add_argument('--step_len', type=float, default=10)
    parser.add_argument('--iter_max', type=int, default=30000)
    parser.add_argument('--clearance', type=float, default=2, help='2 for random_3d.')
    parser.add_argument('--pc_n_points', type=int, default=2048)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_3d', help='random_3d')
    parser.add_argument('--iter_after_initial', type=int, default=5000, help='random_3d use only.')

    parser.add_argument('--num_problems', type=int, help='number of problems to evaluate. None means evaluate all.')
    parser.add_argument('--task_idx', type=str, default=None,
    help='指定任务序号，比如 "10" 或 "1,5,8" 或 "0-9"')

    return parser.parse_args()



args = arg_parse()
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {args.device}")
# * sanity check
if args.path_planner == 'rrt_star' or args.path_planner == 'irrt_star':
    assert args.neural_net == 'none'
else:
    assert args.neural_net != 'none'
#  * set get_path_planner
if args.neural_net == 'none':
    path_planner_name = args.path_planner
elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
    path_planner_name = args.path_planner+'_png'
elif args.neural_net == 'unet':
    path_planner_name = args.path_planner+'_gng'
else:
    raise NotImplementedError
if args.connect != 'none':
    path_planner_name = path_planner_name+'_c'
path_planner_name = path_planner_name+'_3d'
get_path_planner = getattr(import_module('path_planning_classes_3d.'+path_planner_name), 'get_path_planner')
#  * set NeuralWrapper
if args.neural_net == 'none':
    NeuralWrapper = None
elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
    neural_wrapper_name = args.neural_net+'_wrapper'
    if args.connect != 'none':
        neural_wrapper_name = neural_wrapper_name+'_connect_'+args.connect
    NeuralWrapper = getattr(import_module('wrapper_3d.pointnet_pointnet2.'+neural_wrapper_name), 'PNGWrapper')
else:
    raise NotImplementedError
#  * set planning problem
get_env_configs = getattr(import_module('datasets_3d.planning_problem_utils_3d'), 'get_'+args.problem+'_env_configs')
get_problem_input = getattr(import_module('datasets_3d.planning_problem_utils_3d'), 'get_'+args.problem+'_problem_input')

# * main
if NeuralWrapper is None:
    neural_wrapper = None
else:
    neural_wrapper = NeuralWrapper(
        device=args.device, coord_dim=3
    )
if args.problem == 'random_3d':
    args.clearance = 2
print(args)
env_config_list = get_env_configs()
if args.num_problems is None:
    num_problems = len(env_config_list)
else:
    assert args.num_problems <= len(env_config_list)
    num_problems = args.num_problems
result_folderpath = 'results/evaluation/3d'
makedirs(result_folderpath, exist_ok=True)

if args.connect != 'none':
    connect_str = '-c-'+args.connect
else:
    connect_str = ''
eval_setting = args.problem+'-'+args.path_planner+connect_str+'-'+args.neural_net+'-'+str(num_problems)
result_filepath = join(result_folderpath, eval_setting+'.pickle')
if not exists(result_filepath):
    env_result_config_list = []
else:
    with open(result_filepath, 'rb') as f:
        env_result_config_list = pickle.load(f)

# 解析任务索引
if args.task_idx is not None:
    task_indices = []
    for part in args.task_idx.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            task_indices.extend(range(start, end+1))
        else:
            task_indices.append(int(part))
    task_indices = sorted(set(task_indices))
    env_config_list = [env_config_list[i] for i in task_indices]
    num_problems = len(env_config_list)

# 初始化进度条
env_result_config_list = [] if not exists(result_filepath) else pickle.load(open(result_filepath, 'rb'))
num_problems_to_eval = num_problems - len(env_result_config_list)

with tqdm(total=num_problems_to_eval, desc=f'Evaluating {eval_setting}', unit='problem') as pbar:
    for env_idx, env_config in enumerate(env_config_list[:num_problems]):
        if env_idx < len(env_result_config_list):
            pbar.update(1)
            continue

        problem = get_problem_input(env_config)
        path_planner = get_path_planner(
            args,
            problem,
            neural_wrapper,
        )
        print
        if args.problem == 'random_3d':
            path_len_list, first_solution_time, total_planning_time = path_planner.planning_random(iter_after_initial=5000)
            print("Time to first solution:", first_solution_time)
            print("Total planning time:", total_planning_time)

            plt.plot(range(len(path_len_list)), path_len_list)
            plt.xlabel("Iteration")
            plt.ylabel("Best Path Length")
            plt.title("Convergence Curve (planning_random)")
            plt.tight_layout()

            plt.savefig(f"results/evaluation/3d/convergence_curve_env{env_idx}.png")
            plt.close()
        else:
            raise NotImplementedError

        env_result_config = copy(env_config)
        env_result_config['result'] = path_len_list
        env_result_config_list.append(env_result_config)

        with open(result_filepath, 'wb') as f:
            pickle.dump(env_result_config_list, f)

        pbar.update(1)  # 更新进度条
