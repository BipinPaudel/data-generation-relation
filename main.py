from src.configs.config import Experiment
from src.utils.initialization import read_config_from_yaml
import argparse
from src.service import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument('--experiment', type=str, default=[Experiment.PSEUDO_LABEL_GENERATION.value], help='The type of experiment')
    parser.add_argument('--dataset', type=str, default='semeval', help='type of dataset')
    parser.add_argument('--data_type', type=str, default='test', help='type of dataset')
    args = parser.parse_args()
    
    env = "configs/reddit_llama3.1_70b.yaml"
    env = "configs/quen72b.yaml"
    cfg = read_config_from_yaml(env)
    data_type = ['test']
    for exp in args.experiment:
        for dtype in data_type:
            run_experiment(cfg, exp, args.dataset, dtype)
    
    