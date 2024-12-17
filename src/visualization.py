import io
import time
from enum import Enum
from typing import Tuple, Union

import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from trimesh_utils import get_icosphere, asSpherical


class ViewPoint(str, Enum):
    top = "top"
    side1 = "side1"
    side2 = "side2"
    side3 = "side3"
    side4 = "side4"
    bottom = "bottom"


viewpoint_map = {
    "top": np.array([   [1,0,0,0],      [0,1,0,0],      [0,0,1,np.pi],  [0,0,0,1]]),
    "side1": np.array([ [1,0,0,0],      [0,0,-1,-np.pi],[0,1,0,0],      [0,0,0,1]]),
    "side2": np.array([ [0,0,1,np.pi],  [1,0,0,0],      [0,1,0,0],      [0,0,0,1]]),
    "side3": np.array([ [-1,0,0,0],     [0,0,1,np.pi],  [0,1,0,0],      [0,0,0,1]]),
    "side4": np.array([ [0,0,-1,-np.pi],[-1,0,0,0],     [0,1,0,0],      [0,0,0,1]]),
    "bottom": np.array([[1,0,0,0],      [0,-1,0,0],     [0,0,-1,-np.pi],[0,0,0,1]]),
}


def rotated_viewpoint(t):
    """# t: theta 0->2pi"""
    return np.array([
        [np.cos(t), 0,  np.sin(t),  np.pi * np.sin(t)],
        [np.sin(t), 0,  -np.cos(t), -np.pi * np.cos(t)],
        [0,1,0,0],
        [0,0,0,1]]
    )


class SphereVisualizer:

    def __init__(self, rank: int, node_type: str, depth_color_map: str = None, depth_invert: bool = False, sem_colors: np.ndarray = None):
        assert node_type in ("face", "vertex")

        self.rank = rank
        self.node_type = node_type

        self.mesh = get_icosphere(subdivisions=rank)
        self.scene = self.mesh.scene()

        face_normals = self.mesh.face_normals
        face_normals_rphitheta = asSpherical(face_normals)
        self.face_normals_wh = np.stack(
            (
                face_normals_rphitheta[:, 2] / 360 * 2,
                face_normals_rphitheta[:, 1] / 180 * 2 - 1,
            ),
            axis=1,
        ).astype(np.float32)

        vertex_normals = self.mesh.vertices
        vertex_normals_rphitheta = asSpherical(vertex_normals)
        self.vertex_normals_wh = np.stack(
            (
                vertex_normals_rphitheta[:, 2] / 360 * 2,
                vertex_normals_rphitheta[:, 1] / 180 * 2 - 1,
            ),
            axis=1,
        ).astype(np.float32)

        self.depth_color_map = mpl.colormaps[depth_color_map] if depth_color_map is not None else None
        self.depth_invert = depth_invert
        self.sem_colors = sem_colors

    def reset_mesh(self):
        face_colors = self.mesh.visual.face_colors
        self.mesh = get_icosphere(subdivisions=self.rank)
        self.scene = self.mesh.scene()
        self.mesh.visual.face_colors = face_colors

    def set_viewpoint(self, viewpoint: Union[np.ndarray, ViewPoint]):
        if isinstance(viewpoint, np.ndarray):
            self.scene.camera_transform = viewpoint.copy()
        else:
            self.scene.camera_transform = viewpoint_map[viewpoint]

    def render(self, resolution: Tuple[int, int], smooth: bool = False):
        success = False
        for i in range(5):
            try:
                raw_image = self.scene.save_image(resolution=resolution, visible=True, smooth=smooth)
                success = True
                break
            except ZeroDivisionError as e:
                print(f"Failed scene.save_image {i+1} times..")
                time.sleep(0.5)
        if not success:
            raise RuntimeError("Failed scene.save_image too many times (5)")
        if i > 0:
            print(f"Succeeded scene.save_image at attempt {i + 1}.")
            self.reset_mesh()
        time.sleep(0.2)
        image = np.array(Image.open(io.BytesIO(raw_image)))
        return image

    def put_rgb_data(self, data: np.ndarray):
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)
        assert data.ndim == 2
        assert data.shape[0] == self.mesh.faces.shape[0]
        assert data.shape[1] in (1, 3)
        assert data.dtype == "uint8"

        self.mesh.visual.face_colors[:, :3] = data

    def get_rgb_image(self, data: np.ndarray, resolution: Tuple[int, int], viewpoint: Union[np.ndarray, ViewPoint], smooth: bool = False):
        self.put_rgb_data(data)
        self.set_viewpoint(viewpoint)
        return self.render(resolution, smooth)

    def put_depth_data(self, data: np.ndarray, valid_mask: np.ndarray):
        if data.ndim == 2:
            data = np.squeeze(data, 1)
        if valid_mask.ndim == 2:
            valid_mask = np.squeeze(valid_mask, 1)
        assert data.ndim == 1, "data.ndim == 1"
        assert valid_mask.shape == data.shape, "valid_mask.shape == data.shape"
        assert data.shape[0] == self.mesh.faces.shape[0], "data.shape[0] == self.mesh.faces.shape[0]"
        assert data.dtype == "uint8", "data.dtype == uint8"

        face_colors = self.depth_color_map(data if not self.depth_invert else 1 - data)
        face_colors[valid_mask==0, :3] = 0
        self.mesh.visual.face_colors = face_colors

    def get_depth_image(self, data: np.ndarray, valid_mask: np.ndarray, resolution: Tuple[int, int], viewpoint: Union[np.ndarray, ViewPoint], smooth: bool = False):
        self.put_depth_data(data, valid_mask)
        self.set_viewpoint(viewpoint)
        return self.render(resolution, smooth)

    def put_semantic_data(self, data: np.ndarray, valid_mask: np.ndarray):
        if data.ndim == 2:
            data = np.squeeze(data, 1)
        if valid_mask.ndim == 2:
            valid_mask = np.squeeze(valid_mask, 1)
        assert data.ndim == 1
        assert valid_mask.shape == data.shape
        assert data.shape[0] == self.mesh.faces.shape[0]

        face_colors = self.sem_colors[data]
        face_colors[valid_mask == 0, :3] = 0
        self.mesh.visual.face_colors = face_colors

    def get_sem_image(self, data: np.ndarray, valid_mask: np.ndarray, resolution: Tuple[int, int], viewpoint: Union[np.ndarray, ViewPoint], smooth: bool = False):
        self.put_sem_data(data, valid_mask)
        self.set_viewpoint(viewpoint)
        return self.render(resolution, smooth)

    def vertices_to_faces(self, vertices_sphere_image: np.ndarray) -> np.ndarray:
        if self.node_type == "face":
            return vertices_sphere_image

        face_vertices = self.mesh.faces

        faces_sphere_image = np.stack((vertices_sphere_image[face_vertices[:, 0]], vertices_sphere_image[face_vertices[:, 1]], vertices_sphere_image[face_vertices[:, 2]]), axis=1)
        faces_sphere_image = faces_sphere_image.mean(1)

        return faces_sphere_image

    def mask_vertices_to_faces(self, vertices_sphere_valid_mask: np.ndarray) -> np.ndarray:
        if self.node_type == "face":
            return vertices_sphere_valid_mask

        face_vertices = self.mesh.faces

        faces_sphere_valid_mask = np.stack((vertices_sphere_valid_mask[face_vertices[:, 0]], vertices_sphere_valid_mask[face_vertices[:, 1]], vertices_sphere_valid_mask[face_vertices[:, 2]]), axis=1)
        faces_sphere_valid_mask = faces_sphere_valid_mask.all(1)

        return faces_sphere_valid_mask
