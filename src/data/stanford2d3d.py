import os
import random
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms

from trimesh_utils import get_icosphere, asSpherical, IcoSphereRef


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class MISSING:
    pass


class CorruptedDataError(Exception):
    pass


class Stanford2D3D(data.Dataset):
    """The Stanford2D3D Dataset"""
    MAX_DEPTH: int = 5120
    NUM_CLASSES: int

    def __init__(
            self,
            root_dir,
            list_file,
            dataset_kwargs,
            augmentation_kwargs: Optional[Dict] = None,
            is_training: bool = MISSING,
    ):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_lr_flip_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        assert not isinstance(is_training, MISSING)

        super().__init__()
        self.id2label = np.load("./data/stanford2d3d_id2label.npy")
        self.sem_colors = np.load("./data/stanford2d3d_colors.npy")
        self.NUM_CLASSES = len(self.sem_colors)

        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.valid_mask = cv2.imread("./data/stanford2d3d_mask_pretty.png", -1) > 0

        self.sphere_rank = dataset_kwargs["sphere_rank"]
        self.grid_width = dataset_kwargs["grid_width"]
        self.sphere_node_type = dataset_kwargs["sphere_node_type"]

        self.icosphere_ref = IcoSphereRef(self.sphere_node_type)

        normals = self.icosphere_ref.get_normals(rank=self.sphere_rank)
        normals_rphitheta = asSpherical(normals)
        self.normals_wh = np.stack(
            (
                normals_rphitheta[:, 2] / 180,  # [-180, 180] -> [-1, 1]
                normals_rphitheta[:, 1] / 180 * 2 - 1,  # [0, 180] -> [-1, 1]
            ),
            axis=1,
        ).astype(np.float32)

        if augmentation_kwargs is not None:
            self.color_augmentation = augmentation_kwargs["color_augmentation"]
            self.lr_flip_augmentation = augmentation_kwargs["lr_flip_augmentation"]
            self.yaw_rotation_augmentation = augmentation_kwargs["yaw_rotation_augmentation"]
        else:
            self.color_augmentation = False
            self.lr_flip_augmentation = False
            self.yaw_rotation_augmentation = False

        self.is_training = is_training

        self.to_tensor = transforms.ToTensor()

        self.norm_mean = 0.5
        self.norm_std = 0.225

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx, attempt=1):
        try:
            return self.getitem(idx)
        except CorruptedDataError as e:
            print(e)
            print(f"Failed (TRY {attempt}) loading image {idx}: {self.rgb_depth_list[idx][0]}")
            if self.is_training:
                if attempt <= 5:
                    new_idx = random.randint(0, len(self)-1)
                    return self.__getitem__(new_idx, attempt+1)
            raise e

    def getitem(self, idx):
        inputs = {
            # "orig_rgb": None,
            # "orig_gt_depth": None,
            # "orig_masked_gt_depth": None,

            # "augmented_orig_rgb": None,
            # "augmented_orig_gt_depth": None,

            "grid_rgb": None,
            "grid_gt_depth": None,
            "grid_gt_sem": None,
            "sphere_rgb": None,
            "sphere_gt_depth": None,
            "sphere_gt_sem": None,

            "normalized_grid_rgb": None,
            "normalized_grid_gt_depth": None,
            "normalized_sphere_rgb": None,
            "normalized_sphere_gt_depth": None,
        }

        rgb_name = os.path.join(self.root_dir, "stanford2d3d", self.rgb_depth_list[idx][0])
        depth_name = os.path.join(self.root_dir, "stanford2d3d", self.rgb_depth_list[idx][1])
        sem_name = os.path.join(self.root_dir, "stanford2d3d", self.rgb_depth_list[idx][1].replace("depth", "semantic"))
        assert os.path.isfile(rgb_name)
        assert os.path.isfile(depth_name)
        assert os.path.isfile(sem_name)

        # Load data
        rgb = cv2.imread(rgb_name)
        if rgb is None: raise CorruptedDataError(f"{rgb_name}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        # inputs["orig_rgb"] = rgb.copy()

        gt_depth = cv2.imread(depth_name, -1)
        if gt_depth is None: raise CorruptedDataError(f"{depth_name}")
        gt_depth = gt_depth.astype(np.float32)
        # inputs["orig_gt_depth"] = gt_depth.copy()

        gt_sem = cv2.imread(sem_name)
        if gt_sem is None: raise CorruptedDataError(f"{sem_name}")
        gt_sem = cv2.cvtColor(gt_sem, cv2.COLOR_BGR2RGB)
        gt_sem = self.semantic_to_labels(gt_sem)
        # inputs["orig_gt_sem"] = gt_sem.copy()

        valid_mask = self.valid_mask.copy()
        # inputs["orig_valid_mask"] = valid_mask.copy()

        assert rgb.shape[:2] == gt_depth.shape[:2]
        assert rgb.shape[:2] == gt_sem.shape[:2]
        assert rgb.shape[:2] == valid_mask.shape[:2]

        # Masking
        gt_depth[gt_depth > self.MAX_DEPTH] = np.inf
        gt_depth[gt_depth == 0] = np.inf
        gt_depth[valid_mask == 0] = np.inf
        gt_sem[valid_mask == 0] = 0
        # inputs["orig_masked_gt_depth"] = gt_depth.copy()

        if self.is_training:
            if self.yaw_rotation_augmentation:
                # random yaw rotation
                roll_idx = random.randint(0, rgb.shape[1])
                rgb = np.roll(rgb, roll_idx, axis=1)
                gt_depth = np.roll(gt_depth, roll_idx, axis=1)
                gt_sem = np.roll(gt_sem, roll_idx, axis=1)

            if self.lr_flip_augmentation:
                # random horizontal flip
                if random.random() > 0.5:
                    rgb = np.fliplr(rgb)
                    gt_depth = np.fliplr(gt_depth)
                    gt_sem = np.fliplr(gt_sem)

            if self.color_augmentation:
                rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))

        # inputs["augmented_orig_rgb"] = rgb.copy()
        # inputs["augmented_orig_gt_depth"] = gt_depth.copy()

        grid_rgb, grid_gt_depth, grid_gt_sem = self._convert_to_grid(rgb.copy(), gt_depth.copy(), gt_sem.copy())
        sphere_rgb, sphere_gt_depth, sphere_gt_sem = self._convert_to_sphere(rgb.copy(), gt_depth.copy(), gt_sem.copy())

        inputs["grid_rgb"] = grid_rgb.copy()
        inputs["grid_gt_depth"] = grid_gt_depth.copy()
        inputs["grid_gt_sem"] = grid_gt_sem.copy()
        inputs["sphere_rgb"] = sphere_rgb.copy()
        inputs["sphere_gt_depth"] = sphere_gt_depth.copy()
        inputs["sphere_gt_sem"] = sphere_gt_sem.copy()

        inputs: Dict[str, torch.Tensor] = self._inputs_to_tensors(inputs)

        inputs["grid_gt_depth"] = torch.masked_fill(inputs["grid_gt_depth"], inputs["grid_gt_depth"].isnan(), np.inf)
        inputs["sphere_gt_depth"] = torch.masked_fill(inputs["sphere_gt_depth"], inputs["sphere_gt_depth"].isnan(), np.inf)

        inputs["normalized_grid_rgb"] = (inputs["grid_rgb"] - self.norm_mean) / self.norm_std
        inputs["normalized_sphere_rgb"] = (inputs["sphere_rgb"] - self.norm_mean) / self.norm_std
        inputs["grid_valid_mask"] = (0 < inputs["grid_gt_depth"]) & (inputs["grid_gt_depth"] <= self.MAX_DEPTH)
        inputs["sphere_valid_mask"] = (0 < inputs["sphere_gt_depth"]) & (inputs["sphere_gt_depth"] <= self.MAX_DEPTH)
        inputs["normalized_grid_gt_depth"] = inputs["grid_gt_depth"].clip(0, self.MAX_DEPTH) / self.MAX_DEPTH
        inputs["normalized_sphere_gt_depth"] = inputs["sphere_gt_depth"].clip(0, self.MAX_DEPTH) / self.MAX_DEPTH

        return inputs

    def _inputs_to_tensors(self, inputs):
        return {
            "grid_rgb": torch.tensor(inputs["grid_rgb"]).permute(2,0,1).float() / 255,
            "grid_gt_depth": torch.tensor(inputs["grid_gt_depth"]).float(),
            "grid_gt_sem": torch.tensor(inputs["grid_gt_sem"]),
            "sphere_rgb": torch.tensor(inputs["sphere_rgb"]).float() / 255,
            "sphere_gt_depth": torch.tensor(inputs["sphere_gt_depth"]).float(),
            "sphere_gt_sem": torch.tensor(inputs["sphere_gt_sem"]),
        }

    def _convert_to_grid(self, rgb: np.ndarray, depth: np.ndarray, sem: np.ndarray):
        rgb = cv2.resize(rgb, dsize=(self.grid_width, self.grid_width//2), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, dsize=(self.grid_width, self.grid_width//2), interpolation=cv2.INTER_AREA)
        sem = cv2.resize(sem, dsize=(self.grid_width, self.grid_width//2), interpolation=cv2.INTER_NEAREST)

        return rgb, depth, sem

    def _convert_to_sphere(self, rgb: np.ndarray, depth: np.ndarray, sem: np.ndarray):
        sphere_grid = torch.tensor(self.normals_wh).reshape(1, -1, 1, 2)

        rgb = torch.tensor(rgb).permute(2, 0, 1).float().unsqueeze(0)
        depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
        sem = torch.tensor(sem).unsqueeze(0).unsqueeze(0).float()

        rgb = F.grid_sample(
            input=rgb,
            grid=sphere_grid,
            padding_mode="border",
            align_corners=False,
        )
        rgb = rgb.squeeze(0).squeeze(2).permute(1, 0).reshape(-1, 1, 3).squeeze(1).numpy().astype(np.uint8)


        depth = F.grid_sample(
            input=depth,
            grid=sphere_grid,
            padding_mode="border",
            align_corners=False,
        )
        depth = depth.squeeze(0).squeeze(0).squeeze(1).numpy()

        sem = F.grid_sample(
            input=sem,
            grid=sphere_grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        sem = sem.squeeze(0).squeeze(0).squeeze(1).numpy().astype(np.uint8)

        return rgb, depth, sem

    def semantic_to_labels(self, sem):
        # ASSUMING HERE sem is RGB
        label = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        unk = sem[..., 0] != 0
        label[unk] = 0
        return label
