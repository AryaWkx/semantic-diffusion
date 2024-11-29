import os
import math
import torch
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from utils import image_grid

from matplotlib import pyplot as plt

class RandomProjectionMethod:
    def __init__(self, sd, runtime_seed=42, cache="results/rp/"):
        torch.manual_seed(runtime_seed)
        self.sd = sd
        self.device = sd.device
        self.cache = cache
        os.makedirs(self.cache, exist_ok=True)

    def sample(self, num_samples=100, force_rerun=False, etas=None):
        out_path = self.cache + f"rp-{self.sd.model_label}-samples-{num_samples}-etas-{etas}-raw.pt"
        if os.path.exists(out_path) and not force_rerun:
            hhs = torch.load(out_path)
            print("[INFO] Loaded from", out_path)
            return hhs
        hhs = []
        for _ in tqdm(range(num_samples)):
            hs = self.sd.sample(etas=etas).hs.detach().cpu()[None]
            hhs.append(hs)
        hhs = torch.cat(hhs)
        torch.save(hhs, out_path)
        print("[INFO] Saved to", out_path)
        return hhs

    def get_random_directions_all(self, hhs, num_directions=25):
        num_samples = hhs.shape[0]
        X = hhs.view(num_samples, math.prod(self.sd.diff.h_shape)).float()
        feature_shape = hhs.shape[1:]

        # Generate random directions
        random_directions = torch.randn((num_directions, X.shape[1]), device=X.device)
        random_directions = torch.nn.functional.normalize(random_directions, dim=1)

        # Project data onto random directions
        projections = torch.matmul(X, random_directions.T)
        scaled_directions = projections / (num_samples - 1) ** 0.5
        reshaped_directions = random_directions.reshape((num_directions,) + feature_shape)

        return reshaped_directions, scaled_directions

    def get_random_directions_indv(self, hhs, num_directions=25):
        N, I = hhs.shape[:2]
        individual_feat_shape = hhs.shape[2:]

        hhs_pr_step = (hhs[:, i].reshape(N, math.prod(individual_feat_shape)) for i in range(self.sd.num_inference_steps))

        random_directions = []
        projections = []

        for X in tqdm(hhs_pr_step):
            rnd_dirs = torch.randn((num_directions, X.shape[1]), device=X.device)
            rnd_dirs = torch.nn.functional.normalize(rnd_dirs, dim=1)

            proj = torch.matmul(X, rnd_dirs.T)
            proj = proj / (N - 1) ** 0.5

            random_directions.append(rnd_dirs)
            projections.append(proj[None])

        random_directions = torch.cat([d.reshape((num_directions,) + individual_feat_shape).unsqueeze(1) for d in random_directions], dim=1)
        projections = torch.cat(projections).T

        return random_directions, projections

class RPManipulator(RandomProjectionMethod):
    def __init__(self, sd, num_samples=500, sample_etas=None):
        super().__init__(sd)
        hhs = self.sample(num_samples=num_samples, etas=sample_etas)
        self.random_directions, self.scaled_projections = self.get_random_directions_all(hhs)

    def apply_direction(self, q, dir_idx=0, strength=1):
        hhs = self.random_directions[dir_idx]
        q_edit = self.sd.apply_direction(q, Q(delta_hs=hhs), scale=strength, space="hspace")
        return q_edit
