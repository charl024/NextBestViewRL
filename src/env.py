""" Gym environment, select k views of an object one a time """
from functools import lru_cache
from pathlib import Path 

import gymnasium as gym 
import numpy as np
from gymnasium import spaces

from score import load_cache

REPO = Path(__file__).resolve().parent.parent

def cache_files(cache, max_files=None):
    files = sorted(str(p) for p in (REPO / cache).glob("*.npz"))
    return files[:max_files] if max_files else files

class NBVEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 cache="data/cache/train",
                 files=None,
                 max_files=None,
                 k=4, n_views=100,
                 img=32, reward_scale=10.0,
                 history="all",
                 use_dirs=True,
                 cache_size=128,
                 render_mode=None):
        super().__init__()
        self.files = list(files) if files is not None else cache_files(cache, max_files)
        assert self.files, f"no .npz files in {REPO / cache}"
        self.k = k
        self.N = n_views
        self.img = img

        self.reward_scale = reward_scale
        self.history = history
        self.use_dirs = use_dirs

        self._load = lru_cache(maxsize=cache_size)(load_cache)
        self.n_obs = k * img * img + 3 * k + n_views + 1
        self.observation_space = spaces.Box(-1.0, 1.0, (self.n_obs,), np.float32)
        self.action_space = spaces.Discrete(n_views)

    def _depth(self, a):
        d = self.obj["depth"][a]
        if self.img != d.shape[0]:
            f = d.shape[0] // self.img
            d = d.reshape(self.img, f, self.img, f).mean(axis=(1, 3))
        return d 

    def _add(self, a):
        P = self.obj["P"]
        g = float(np.maximum(P[a] - self.best, 0.0).sum()) / self.obj["F_all"]
        self.best = np.maximum(self.best, P[a])
        self.visited[a] = True
        self.depth[self.t] = self._depth(a)
        u = self.obj["views"][a]
        self.dirs[self.t] = u / np.linalg.norm(u)
        self.chosen.append(int(a))
        self.t += 1
        return g

    def _obs(self):
        depth = self.depth
        dirs = self.dirs
        if self.history == "last":
            depth = np.zeros_like(depth)
            depth[self.t - 1] = self.depth[self.t - 1]
        
        if not self.use_dirs:
            dirs = np.zeros_like(dirs)
        
        return np.concatenate([depth.ravel(), dirs.ravel(), self.visited.astype(np.float32), [self.t / self.k]]).astype(np.float32)

    def _info(self):
        return {"score" : float(self.best.sum()) / self.obj["F_all"]}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.file = options.get("file") or self.files[self.np_random.integers(len(self.files))]
        self.obj = self._load(self.file)
        start = options.get("start")
        start = int(self.np_random.integers(self.N)) if start is None else int(start)
        self.best = np.zeros(self.obj["P"].shape[1], np.float32)
        self.visited = np.zeros(self.N, bool)
        self.depth = np.zeros((self.k, self.img, self.img), np.float32)
        self.dirs = np.zeros((self.k, 3), np.float32)
        self.chosen, self.t = [], 0
        self._add(start)
        return self._obs(), self._info()

    def step(self, action):
        a = int(action)
        
        if self.visited[a]:
            reward = 0.0
            self.depth[self.t] = self._depth(a)
            self.chosen.append(a)
            self.t += 1
        else:
            reward = self.reward_scale * self._add(a)
        
        return self._obs(), reward, self.t >= self.k, False, self._info()

    def action_masks(self):
        return ~self.visited

gym.register(id="NBV-v0", entry_point="env:NBVEnv")
