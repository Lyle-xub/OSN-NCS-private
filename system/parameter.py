import numpy as np
import os
import torch
import math
import random
import pickle
import json
from system.utils import *
from multiprocessing import cpu_count
import argparse


class BaseParams:
    exp_id = "0"
    manual_seed = random.randint(1, 10000)
    use_cuda = True
    max_epoch = 50
    batch_size = 20
    subbatch_size = 20
    lr = 0.01

    def __init__(self):
        if self.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None:
                if hasattr(self, key):
                    setattr(self, key, value)
                elif key in self.core["network"]:  # Check in 'network'
                    self.core["network"][key] = value
                elif key in self.core["data"]:  # check in 'data'
                    self.core["data"][key] = value
                elif "detectors" in self.core["data"] and key in self.core["data"]["detectors"]:  # 嵌套字典
                    self.core["data"]["detectors"][key] = value


class SimParams(BaseParams):

    def __init__(self, model_type="s2nn"):
        self.exp = "test"
        self.exp_id = 0
        self.camera = "event"
        self.model_type = model_type  # here s2nn/s4nn

        self.core = {
            "network": {
                "whole_dim": 700,
                "phase_dim": 400,
                "pixel_size": 12.5e-06,
                "focal_length": 40e-2,
                "wave_lambda": 520e-9,
                "layer_num": 4,
                "intensity_mode": False,
                "scalar": None,
            },
            "data": {
                "load_checkpoint_dir": None,
                "detectors": {
                    "detx": None,
                    "dety": None,
                    "square_size": 40,
                },
            },
        }
        self.loss_slice = slice(
            self.core["network"]["whole_dim"] // 2 - self.core["network"]["phase_dim"] // 2,
            self.core["network"]["whole_dim"] // 2 + self.core["network"]["phase_dim"] // 2,
        )

        self.checkpoint_dir, self.exp_id = create_checkpoint_directory(self.exp)

        canvas_size = (
            self.core["network"]["phase_dim"],
            self.core["network"]["phase_dim"],
        )
        square_size = self.core["data"]["detectors"]["square_size"]
        pattern = [3, 4, 3]
        square_coordinates = generate_square_coordinates(canvas_size, square_size, pattern)

        ordered_coordinates = [
            *square_coordinates[0:3],  # First row
            *square_coordinates[3:7],  # Second row
            *square_coordinates[7:10],  # Third row
        ]
        pad = (self.core["network"]["whole_dim"] - self.core["network"]["phase_dim"]) // 2

        # Setting detector configurations
        self.core["data"]["detectors"]["dety"] = [coord[0] for coord in ordered_coordinates]
        self.core["data"]["detectors"]["detx"] = [coord[1] for coord in ordered_coordinates]
        self.det_x_loc = [coord[1] + pad for coord in ordered_coordinates]
        self.det_y_loc = [coord[0] + pad for coord in ordered_coordinates]
        self.checkpoint_dir = None
        if self.model_type == "s4nn":
            self.core["network"]["whole_dim"] = 400
            self.core["network"]["phase_dim"] = 250
            self.dataset = "b-hc"
        super().__init__()


class ExpParams(BaseParams):

    def __init__(self, model_type="s4nn"):
        self.exp = "exp"
        self.exp_id = 0
        self.train = "bat"
        self.camera = "event"
        self.is_separable = True
        self.unitary = False
        self.fusion = "new"
        self.model_type = model_type

        self.core = {
            "network": {
                "whole_dim": 500,
                "phase_dim": 400,
                "pixel_size": 12.5e-06,
                "focal_length": 40e-2,
                "wave_lambda": 520e-9,
                "layer_num": 4,
                "intensity_mode": False,
                "scalar": None,
            },
            "data": {
                "load_checkpoint_dir": None,
                "detectors": {
                    "detx": None,
                    "dety": None,
                    "square_size": 40,
                },
            },
        }
        canvas_size = (
            self.core["network"]["phase_dim"],
            self.core["network"]["phase_dim"],
        )
        square_size = self.core["data"]["detectors"]["square_size"]
        pattern = [3, 4, 3]
        square_coordinates = generate_square_coordinates(canvas_size, square_size, pattern)

        ordered_coordinates = [
            *square_coordinates[0:3],  # First row
            *square_coordinates[3:7],  # Second row
            *square_coordinates[7:10],  # Third row
        ]
        pad = (self.core["network"]["whole_dim"] - self.core["network"]["phase_dim"]) // 2

        # Setting detector configurations
        self.core["data"]["detectors"]["dety"] = [coord[0] for coord in ordered_coordinates]
        self.core["data"]["detectors"]["detx"] = [coord[1] for coord in ordered_coordinates]
        self.det_x_loc = [coord[1] + pad for coord in ordered_coordinates]
        self.det_y_loc = [coord[0] + pad for coord in ordered_coordinates]

        self.alpha = 0.5
        self.beta = 1.0
        self.loss_slice = slice(
            self.core["network"]["whole_dim"] // 2 - self.core["network"]["phase_dim"] // 2,
            self.core["network"]["whole_dim"] // 2 + self.core["network"]["phase_dim"] // 2,
        )
        self.num_consumers = min(8, cpu_count())
        self.array_size_events = int(1e8)
        self.lr = 1e-2
        self.batch_size = 16
        self.log_batch_num = 10
        self.pic_time = 300
        self.ill_time = 150

        self.checkpoint_dir = None

        if self.model_type == "s4nn":
            self.core["network"]["whole_dim"] = 400
            self.core["network"]["phase_dim"] = 250
            self.core["network"]["layer_num"] = 1
            self.core["network"]["focal_length"] = 30e-2
            self.dataset = "b-hc"
        super().__init__()


def config(type="sim"):
    parser = argparse.ArgumentParser(description="Parameters")

    parser.add_argument("--exp_id", type=str, help="Experiment ID")
    parser.add_argument("--manual_seed", type=int, help="Manual seed")
    parser.add_argument("--use_cuda", type=bool, help="Use CUDA")
    parser.add_argument("--max_epoch", type=int, help="Max epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--subbatch_size",
        type=int,
        default=BaseParams.subbatch_size,
        help="Sub-batch size",
    )
    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument("--exp", type=str, help="Experiment name")
    parser.add_argument("--camera", type=str, help="Camera type")
    parser.add_argument("--model_type", type=str, default="s2nn", help="Model type (s2nn/s3nn/s4nn)")
    # --- core.network
    parser.add_argument("--whole_dim", type=int, help="Whole dimension")
    parser.add_argument("--phase_dim", type=int, help="Phase dimension")
    parser.add_argument("--pixel_size", type=float, default=12.5e-06, help="Pixel size")
    parser.add_argument("--focal_length", type=float, help="Focal length")
    parser.add_argument("--wave_lambda", type=float, default=520e-9, help="Wave lambda")
    parser.add_argument("--layer_num", type=int, help="Number of layers")
    parser.add_argument("--intensity_mode", action="store_true", help="Enable intensity mode")
    parser.set_defaults(intensity_mode=False)
    parser.add_argument("--scalar", type=str, default=None, help="Scalar value")
    # --- core.data
    parser.add_argument("--load_checkpoint_dir", type=str, default=None, help="Load checkpoint dir")
    parser.add_argument("--square_size", type=int, default=40, help="Detector square size")

    # --- ExpParams
    parser.add_argument("--train", type=str, default="bat", help="Training type")
    parser.add_argument("--is_separable", action="store_true", help="Whether the model is separable")
    parser.add_argument("--no_is_separable", action="store_false", dest="is_separable")
    parser.set_defaults(is_separable=True)

    parser.add_argument("--unitary", action="store_true", help="Whether to use unitary constraint")
    parser.add_argument("--no_unitary", action="store_false", dest="unitary")
    parser.set_defaults(unitary=False)
    parser.add_argument("--fusion", type=str, default="new", help="Fusion type")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha value")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta value")
    parser.add_argument("--num_consumers", type=int, default=min(8, cpu_count()), help="Number of consumers")
    parser.add_argument("--array_size_events", type=int, default=int(1e8), help="Array size for events")
    parser.add_argument("--log_batch_num", type=int, help="Log every n batches")
    parser.add_argument("--pic_time", type=int, help="Picture time")
    parser.add_argument("--ill_time", type=int, help="Illumination time")
    parser.add_argument("--dataset", type=str, help="dataset")

    args = parser.parse_args()

    if type == "sim":
        params = SimParams(model_type=args.model_type)
    elif type == "exp":
        params = ExpParams(model_type=args.model_type)
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    params.update_from_args(args)

    return params
