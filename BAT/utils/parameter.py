import argparse
import os
import yaml
from omegaconf import OmegaConf
from utils.utils import create_unique_experiment_folder
import torch

# def create_unique_experiment_folder(phase_prop, train_type):
#     folder_name = f"experiment_{phase_prop}_{train_type}"
#     os.makedirs(folder_name, exist_ok=True)
#     return folder_name


def load_yaml_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found!")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def config():
    parser = argparse.ArgumentParser(description="Parameters")

    parser.add_argument("--config", type=str, help="Path to config YAML file")

    parser.add_argument("--exp_id", type=str, help="Experiment ID")
    parser.add_argument("--manual_seed", type=int, help="Manual seed")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use CUDA")
    parser.add_argument("--max_epoch", type=int, default=5, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--pnn_lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cn_lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--log_batch_num", type=int, default=10, help="Log batch number")

    # --- Optical parameters ---
    parser.add_argument("--whole_dim", type=int, default=700, help="Whole dimension")
    parser.add_argument("--phase_dim", type=int, default=400, help="Phase dimension")
    parser.add_argument("--pixel_size", type=float, default=12.5e-06, help="Pixel size")
    parser.add_argument("--focal_length", type=float, default=0.3, help="Focal length")
    parser.add_argument("--wave_lambda", type=float, default=520e-9, help="Wave lambda")
    parser.add_argument("--layer_num", type=int, default=4, help="Number of layers")
    parser.add_argument("--intensity_mode", action="store_true", help="Enable intensity mode")
    parser.set_defaults(intensity_mode=False)
    parser.add_argument("--scalar", type=str, default=None, help="Scalar value")
    parser.add_argument("--square_size", type=int, default=40, help="Detector square size")

    # --- Experiment parameters ---
    parser.add_argument("--train", type=str, default="bat", help="Training type")
    parser.add_argument("--is_separable", action="store_true", help="Whether the model is separable")
    parser.add_argument("--no_is_separable", action="store_false", dest="is_separable")
    parser.set_defaults(is_separable=True)

    parser.add_argument("--unitary", action="store_true", help="Whether to use unitary constraint")
    parser.add_argument("--no_unitary", action="store_false", dest="unitary")
    parser.set_defaults(unitary=False)

    parser.add_argument("--phase_error", type=float, default=0.28, help="Phase error")
    parser.add_argument("--fusion", type=str, default="new", help="Fusion type")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value")
    parser.add_argument("--beta", type=float, default=1, help="Alpha value")

    # --- Distributed Training ---
    # parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    args = parser.parse_args()

    if args.config:
        yaml_config = load_yaml_config(args.config)
        args = merge_args_with_yaml(args, yaml_config)

    if torch.distributed.get_rank() == 0:
        folder_path, exp_id = create_unique_experiment_folder(args.phase_error, args.train)
        args.exp_id = exp_id

        if args.unitary:
            log_path = os.path.join(folder_path, "unitary.txt")
        elif args.is_separable:
            log_path = os.path.join(folder_path, "separate.txt")
        else:
            log_path = os.path.join(folder_path, "bat.txt")

        config_dict = vars(args)
        config_dict["log_path"] = log_path
        config_dict["folder_path"] = folder_path
        OmegaConf.save(OmegaConf.create(config_dict), os.path.join(folder_path, "arg.yaml"))

    return args


if __name__ == "__main__":
    args = config()
    print(f"Configuration saved in {args}")
