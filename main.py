from smac.env import StarCraft2Env
from common.arguments import get_args, add_env_args
from common.envwrapper import SubprocVecEnv
from common.runner import Runner
import os
os.environ["SC2PATH"] = "/data1/gk/yangchen/StarCraftII4.6.2"


def create_env(args):
    """"Create multiple envs for data collecting."""
    def create_single_env(map_name, seed):
        sc_env = StarCraft2Env(map_name=map_name, seed=seed)
        return sc_env
    env = SubprocVecEnv([create_single_env(args.map_name, i) for i in range(args.n_envs)])
    return env


if __name__ == '__main__':
    args = get_args()
    env = create_env(args)
    args = add_env_args(args, env)

    runner = Runner(args, env)
    runner.train()
