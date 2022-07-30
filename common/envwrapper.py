""""
Mostly based on openAI Baseline codes.
Modified to work with SMAC.
"""
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
import numpy as np


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action, flag = data
            if flag:
                remote.send((0, True, []))
            else:
                reward, terminal, info = env.step(action)
                remote.send((reward, terminal, info))
        elif cmd == 'reset':
            env.reset()
        elif cmd == 'get_state':
            state = env.get_state()
            remote.send(state)
        elif cmd == 'get_obs':
            obs = env.get_obs()
            remote.send(obs)
        elif cmd == 'get_avail_actions':
            avail_actions = env.get_avail_actions()
            remote.send(avail_actions)
        elif cmd == 'get_env_info':
            info = env.get_env_info()
            remote.send(info)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError


class VecEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        super().__init__()
        self.waiting =False
        self.closed = False
        self.n_envs = len(env_fns)
        self.terminal = [False for _ in range(self.n_envs)]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        # info = self.get_env_info()
        # self.state_shape = info['state_shape']
        # self.obs_shape = info['obs_shape']
        # self.n_agents = info['n_agents']
        # self.n_actions = info['n_actions']

    def reset(self):
        self.terminal = [False for _ in range(self.n_envs)]
        for remote in self.remotes:
            remote.send(('reset', None))

    def step_async(self, actions):
        for remote, action, terminal in zip(self.remotes, actions, self.terminal):
            remote.send(('step', (action, terminal)))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, terminals, infos = zip(*results)
        self.terminal = terminals
        return np.stack(rewards), np.stack(terminals), infos

    def get_env_info(self):
        remote = self.remotes[0]
        remote.send(('get_env_info', None))
        info = remote.recv()
        return info

    def get_state(self):
        for remote in self.remotes:
            remote.send(('get_state', None))
        results = [remote.recv() for remote in self.remotes]
        # for i in range(self.n_envs):
        #     if self.terminal[i]:
        #         results[i] = np.zeros([self.state_shape])
        state = np.stack(results)
        return state

    def get_obs(self):
        for remote in self.remotes:
            remote.send(('get_obs', None))
        results = [remote.recv() for remote in self.remotes]
        # for i in range(self.n_envs):
        #     if self.terminal[i]:
        #         results[i] = np.zeros([self.n_agents, self.obs_shape])
        obs = np.stack(results)
        return obs

    def get_avail_actions(self):
        for remote in self.remotes:
            remote.send(('get_avail_actions', None))
        results = [remote.recv() for remote in self.remotes]
        # for i in range(self.n_envs):
        #     if self.terminal[i]:
        #         results[i] = np.zeros([self.n_agents, self.n_actions])
        #         results[i][:, 0] = 1
        avail_actions = np.stack(results)
        return avail_actions

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_terminal_flag(self):
        flag = True
        for terminal in self.terminal:
            flag *= terminal
        return flag
