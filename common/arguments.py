import argparse


def get_args():
    """"
    Common configuration hyper-parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='qmix', help='algorithm name for training')
    parser.add_argument('--n_envs', type=int, default=8, help='num of envs running in parallel')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads for transformer')
    parser.add_argument('--map_name', type=str, default='3m', help='SMAC map name')
    parser.add_argument('--n_episodes', type=int, default=50000, help='total running episodes')
    parser.add_argument('--buffer_size', type=int, default=5000, help='buffer size counted by episode')
    parser.add_argument('--mini_batch_size', type=int, default=32, help='batch size counted by episode')
    parser.add_argument('--q_steps', type=int, default=1, help='num of q steps')
    parser.add_argument('--a_steps', type=int, default=1, help='num of actor steps')
    parser.add_argument('--evaluate_cycle', type=int, default=100, help='evaluate cycle')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='evaluate epoch counted by episode')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='target net update cycle')
    parser.add_argument('--hidden_shape', type=int, default=64, help='shape of hidden state in NN')
    parser.add_argument('--eps', type=float, default=1.0, help='initial epsilon set for e-greedy')
    parser.add_argument('--min_eps', type=float, default=0.05, help='minimum epsilon set for e-greedy')
    parser.add_argument('--delta_eps', type=float, default=1e-5, help='delta epsilon set for decaying e')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='parameter set for time-decaying reward')
    parser.add_argument('--tau', type=float, default=0.1, help='parameter set for soft updating target net')
    parser.add_argument('--td_lambda', type=float, default=0.1, help='parameter set for TD lambda')
    parser.add_argument('--grad_norm_clip', type=int, default=20, help='clipped gradient')
    parser.add_argument('--device', type=str, default='cuda:0', help='device on running')

    args = parser.parse_args()
    return args


def add_env_args(args, env):
    """"
    Add specific map information.
    """
    info = env.get_env_info()

    args.state_shape = info['state_shape']
    args.obs_shape = info['obs_shape']
    args.n_agents = info['n_agents']
    args.n_actions = info['n_actions']
    args.episode_limit = info['episode_limit']

    return args
