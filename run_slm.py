"""Launch SLM-Lab with the NBV environment and network registered. Use exactly like `slm-lab`:
    cd SLM-Lab && uv run python ../run_slm.py run -s seed=11 nbv/nbv.yaml nbv_full train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import env
import slm_lab.agent.net as slm_nets
from slm_net import NBVConvNet

slm_nets.NBVConvNet = NBVConvNet

from slm_lab.cli import cli

if __name__ == "__main__":
    cli()