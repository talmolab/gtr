from dreem.inference import track
from omegaconf import OmegaConf
import os

# /Users/mustafashaikh/dreem/dreem/training
# /Users/main/Documents/GitHub/dreem/dreem/training
# os.chdir("/Users/main/Documents/GitHub/dreem/dreem/training")
config = "/root/vast/mustafa/dreem-experiments/run/lysosome-baselines/debug/configs/inference.yaml"

cfg = OmegaConf.load(config)

track.run(cfg)