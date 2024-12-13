from dreem.training import train
from omegaconf import OmegaConf
import os

# /Users/mustafashaikh/dreem/dreem/training
# /Users/main/Documents/GitHub/dreem/dreem/training
# os.chdir("/Users/main/Documents/GitHub/dreem/dreem/training")
base_config = "/root/vast/mustafa/dreem-experiments/run/lysosome-baselines/debug/configs/base-updated.yaml"
params_config = "/root/vast/mustafa/dreem-experiments/run/lysosome-baselines/debug/configs/override-updated.yaml"

cfg = OmegaConf.load(base_config)
# Load and merge override config
override_cfg = OmegaConf.load(params_config)
cfg = OmegaConf.merge(cfg, override_cfg)

train.run(cfg)