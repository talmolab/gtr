from dreem.training import train
from omegaconf import OmegaConf

# /Users/mustafashaikh/dreem/dreem/training
# /Users/main/Documents/GitHub/dreem/dreem/training


inference_config = "tests/configs/inference.yaml"

cfg = OmegaConf.load(inference_config)

eval.run(cfg)