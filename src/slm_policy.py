"""Load a trained SLM-Lab PPO network and act with it outside SLM-Lab."""
import glob
import json
from pathlib import Path

import numpy as np
import torch

from slm_net import NBVConvNet

DEV_FRAMES = 20000


def load_spec(run_dir):
    with open(next(Path(run_dir).glob("*_t0_spec.json"))) as fh:
        return json.load(fh)


class SLMPolicy:
    def __init__(self, run_dir, ckpt="final", device="cpu"):
        run_dir = Path(run_dir)
        self.spec = load_spec(run_dir)
        net_spec = dict(self.spec["agent"]["net"], gpu=(device != "cpu"))
        env = self.spec["env"]
        k, img, n_views = env.get("k", 4), env.get("img", 32), env.get("n_views", 100)
        in_dim = k * img * img + 3 * k + n_views + 1
        out_dim = [n_views, 1] if net_spec["shared"] else n_views
        self.net = NBVConvNet(net_spec, in_dim, out_dim)
        tag = "_ckpt-best" if ckpt == "best" else ""
        model_path = next((run_dir / "model").glob(f"*_t0_s0{tag}_net_model.pt"))
        self.net.load_state_dict(torch.load(model_path, map_location=self.net.device))
        self.net.eval()

    def logits(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.net.device).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out[0].cpu().numpy()

    def act(self, obs, mask):
        ''' Most likely view among those not chosen yet. '''
        lg = self.logits(obs)
        lg[~mask] = -np.inf
        return int(lg.argmax())


def final_runs(pattern):
    ''' Run folders matching pattern, excluding dev-mode runs (max_frame <= DEV_FRAMES). '''
    out = []
    for d in sorted(glob.glob(pattern)):
        if load_spec(d)["env"]["max_frame"] > DEV_FRAMES:
            out.append(d)
        else:
            print(f"skipping dev run {d}")
    return out
