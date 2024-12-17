from copy import deepcopy
from typing import List, Union, Tuple, Dict, Set

import numpy as np
import trimesh
import trimesh.creation
from trimesh import Trimesh


def asCartesian(rphitheta: np.ndarray) -> np.ndarray:
    """
    r - radius
    phi - vertical angle [0, 180]
    theta - horizontal angle [-180, 180]

    x,y,z - coordinates on the sphere
    """
    #takes list rthetaphi (single coord)
    r       = rphitheta[0]
    phi     = rphitheta[1] * np.pi/180 # to radian
    theta   = rphitheta[2] * np.pi/180
    x = r * np.sin( phi ) * np.cos( theta )
    y = r * np.sin( phi ) * np.sin( theta )
    z = r * np.cos( phi )
    return [x,y,z]


def asSpherical(xyz: np.ndarray) -> np.ndarray:
    """
    x,y,z - coordinates on the sphere

    r - radius
    phi - vertical angle [0, 180]
    theta - horizontal angle [-180, 180]
    """
    #takes list xyz (single coord)
    x       = xyz[:,0]
    y       = xyz[:,1]
    z       = xyz[:,2]
    r       = np.sqrt(x*x + y*y + z*z)
    phi     = np.arccos(z/r) * 180/np.pi #to degrees
    theta   = np.arctan2(y,x) * 180/np.pi
    return np.stack([r,phi,theta], 1)


def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


def get_icosphere(subdivisions: int, refine=True, radius=1.0, **kwargs):
    """
    Create an isophere centered at the origin.

    Parameters
    ----------
    subdivisions : int
      How many times to subdivide the mesh.
      Note that the number of faces will grow as function of
      4 ** subdivisions, so you probably want to keep this under ~5
    radius : float
      Desired radius of sphere
    color: (3,) float or uint8
      Desired color of sphere

    Returns
    ---------
    ico : trimesh.Trimesh
      Meshed sphere
    """
    def refine_spherical(_mesh):
        vectors = _mesh.vertices
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        _mesh.vertices += unit * offset.reshape((-1, 1))

    ico = trimesh.creation.icosahedron()
    ico._validate = False

    for j in range(subdivisions):
        ico = ico.subdivide()
        if refine:
            refine_spherical(ico)

    return trimesh.Trimesh(
        vertices=ico.vertices,
        faces=ico.faces,
        metadata={'shape': 'sphere',
                  'radius': radius},
        process=kwargs.pop('process', False),
        **kwargs,
    )


def find_face_neighbors(mesh: Trimesh, depth: int) -> List[Set[int]]:

    num_faces = mesh.faces.shape[0]

    old_neighbors = [set() for _ in range(num_faces)]
    neighbors = [{i} for i in range(num_faces)]

    first_order_neighbors = [{f for v in mesh.faces[n] for f in mesh.vertex_faces[v] if f != -1} for n in range(num_faces)]

    for d in range(depth):
        new_neighbors = deepcopy(neighbors)

        for i in range(num_faces):
            for n in neighbors[i] - old_neighbors[i]:
                new_neighbors[i].update(first_order_neighbors[n])

        old_neighbors = neighbors
        neighbors = new_neighbors

    return neighbors


def find_vertex_neighbors(mesh: Trimesh, depth: int) -> List[Set[int]]:

    num_vertices = mesh.vertices.shape[0]

    old_neighbors = [set() for _ in range(num_vertices)]
    neighbors = [{i} for i in range(num_vertices)]

    first_order_neighbors = [set(mesh.vertex_neighbors[n]) for n in range(num_vertices)]

    for d in range(depth):
        new_neighbors = deepcopy(neighbors)

        for i in range(num_vertices):
            for n in neighbors[i] - old_neighbors[i]:
                new_neighbors[i].update(first_order_neighbors[n])

        old_neighbors = neighbors
        neighbors = new_neighbors

    return neighbors


class IcoSphereRef:
    def __init__(self, node_type: str):
        assert node_type in ("face", "vertex")
        self.node_type = node_type

        self.icospheres: Dict[Tuple[int, bool], Trimesh] = {}
        self.neighbor_maps: Dict[Tuple[int, int], List[Set[int]]] = {}

    def get_icosphere(self, rank: int, refine: bool) -> Trimesh:
        if (rank, refine) not in self.icospheres.keys():
            self.icospheres[(rank, refine)] = get_icosphere(subdivisions=rank, refine=refine)
        return self.icospheres[(rank, refine)]

    def get_neighbor_mapping(self, rank: int, depth: int) -> List[Set[int]]:
        if (rank, depth) not in self.neighbor_maps.keys():
            ico = self.get_icosphere(rank=rank, refine=True)
            print(f"Building neighbor mapping {rank}-{depth}")
            if self.node_type == "face":
                self.neighbor_maps[(rank, depth)] = find_face_neighbors(ico, depth=depth)
            elif self.node_type == "vertex":
                self.neighbor_maps[(rank, depth)] = find_vertex_neighbors(ico, depth=depth)
            else:
                raise NotImplementedError(f"Unsupported node type {self.node_type}")
            print(f"Building neighbor mapping Rank:{rank} Depth:{depth} -- DONE")
        return self.neighbor_maps[(rank, depth)]

    def get_normals(self, rank) -> np.ndarray:
        ico = self.get_icosphere(rank, True)

        if self.node_type == "face":
            return ico.face_normals.copy()
        elif self.node_type =="vertex":
            return ico.vertices.copy()
        else:
            raise NotImplementedError(f"Unsupported node type {self.node_type}")
