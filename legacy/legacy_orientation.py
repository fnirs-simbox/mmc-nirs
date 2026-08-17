from __future__ import annotations

import pmmc
import numpy as np
import matplotlib.pyplot as plt
import iso2mesh as i2m
import trimesh
from scipy.spatial import Delaunay

import itertools

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import os
import math
 

def make_to_ras_dict():

    """

    Returns a dict mapping every valid 3-letter orientation code (48 total)

    to a 3x3 matrix (as nested lists) that converts coordinates from that

    orientation into RAS coordinates.

    For more information on RAS coordinate system, check here:

    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm

 

    Convention:

      - The three letters indicate the world axis along which the i, j, k axes increase.

      - 'R'/'L' -> ±X, 'A'/'P' -> ±Y, 'S'/'I' -> ±Z.

      - The returned matrix M satisfies: [x_RAS, y_RAS, z_RAS]^T = M @ [i, j, k]^T.

    """

    # For each world axis (X, Y, Z), the (pos, neg) letter pair:

    axis_letters = [

        ('R', 'L'),  # X

        ('A', 'P'),  # Y

        ('S', 'I'),  # Z

    ]

 

    to_ras = {}

 

    # Permute which world axis each of (i, j, k) aligns to

    for order in itertools.permutations([0, 1, 2], 3):

        # For each axis pick a sign: 0 => positive (R/A/S), 1 => negative (L/P/I)

        for signs in itertools.product([0, 1], repeat=3):

            # Build the orientation code (e.g., 'LPS', 'RAS', ...)

            code_letters = []

            for col in range(3):  # columns correspond to i, j, k axes

                world_axis = order[col]          # which of X/Y/Z

                sign_idx   = signs[col]          # 0 => +, 1 => -

                code_letters.append(axis_letters[world_axis][sign_idx])

            code = ''.join(code_letters)

 

            # Build the 3x3 transform matrix columns (unit vectors in RAS basis)

            # Column c is the RAS vector for axis c (i=0, j=1, k=2)

            M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            for col in range(3):

                world_axis = order[col]  # 0->X, 1->Y, 2->Z

                sign = +1 if signs[world_axis] == 0 else -1

                M[world_axis][col] = sign  # place ±1 in the appropriate row

            to_ras[code] = M

 

    return to_ras

 

to_ras = make_to_ras_dict()

 

# --- Example checks ---

# Identity for RAS:

# to_ras['RAS'] -> [[1,0,0],[0,1,0],[0,0,1]]

# DICOM LPS to RAS (flip X and Y):

# to_ras['LPS'] -> [[-1,0,0],[0,-1,0],[0,0,1]]

# ARS (i=+Y, j=+X, k=+Z) permutes X and Y:

# to_ras['ARS'] -> [[0,1,0],[1,0,0],[0,0,1]]

def register_probe(source_coords, detector_coords, mesh_nodes, mesh_elem, scalp_idx, probe_orientation='RAS', probe_units='mm'):
    #TODO needs docstring
    
    #converting probe coords to mm
    if probe_units == 'mm':
        pass
    elif probe_units == 'm':
        source_coords = source_coords * 1000
        detector_coords = detector_coords * 1000
    else:
        raise ValueError("`probe_units` must be in m or mm")
    
    to_ras = make_to_ras_dict()

    #loading relevant transformation matrix from lookup table
    #load lookup table instead of creating it every time
    transformation_matrix = to_ras[probe_orientation]
    
    #reorienting optode coords to RAS via selected transformation matrix
    num_src = source_coords.shape[0]
    probe_pts = np.vstack([source_coords, detector_coords])
    RAS_probe_coords = probe_pts @ transformation_matrix

    #roughly align optode positions with head mesh
    mesh_shift = np.array([(mesh_nodes.max(0)[0]+mesh_nodes.min(0)[0])/2, mesh_nodes.max(0)[1], mesh_nodes.max(0)[2]])
    probe_shift = np.array([(RAS_probe_coords.max(0)[0]+RAS_probe_coords.min(0)[0])/2,RAS_probe_coords.max(0)[1],RAS_probe_coords.max(0)[2]])


    aligned_probe_coords = RAS_probe_coords - probe_shift + mesh_shift

    reg_source_coords = aligned_probe_coords[:num_src,:]
    reg_detector_coords = aligned_probe_coords[num_src:,:]

    #finding directions of each optode for mmc
    source_directions = find_optode_directions(reg_source_coords,mesh_nodes)
    detector_directions = find_optode_directions(reg_detector_coords,mesh_nodes)

    #finding initial enclosing elements of each optode
    #TODO turn while loop code below into a method
    in_src_mask, _ = points_in_tetrahedral_mesh(reg_source_coords,mesh_elem,mesh_nodes)
    while np.any(in_src_mask):
        in_mask = in_src_mask == True
        reg_source_coords[in_mask] += source_directions[in_mask] * -1
        in_src_mask, _ = points_in_tetrahedral_mesh(reg_source_coords,mesh_elem,mesh_nodes)
    
    in_det_mask, _ = points_in_tetrahedral_mesh(reg_detector_coords,mesh_elem,mesh_nodes)
    while np.any(in_det_mask):
        in_mask = in_det_mask == True
        reg_detector_coords[in_mask] += detector_directions[in_mask] * -1
        in_det_mask, _ = points_in_tetrahedral_mesh(reg_detector_coords,mesh_elem,mesh_nodes)

    
    #iteratively embedding all sources beneath the mesh surface
    while np.any(~in_src_mask):
        out_mask = in_src_mask == False
        reg_source_coords[out_mask] += source_directions[out_mask] * 1
        in_src_mask, init_id_src = points_in_tetrahedral_mesh(reg_source_coords,mesh_elem,mesh_nodes)
    #reg_source_coords[out_mask] += source_directions[out_mask] * 0.1

    #iteratively embedding all detectors beneath the mesh surface
    while np.any(~in_det_mask):
        out_mask = in_det_mask == False
        reg_detector_coords[out_mask] += detector_directions[out_mask] * 1
        in_det_mask, init_id_det = points_in_tetrahedral_mesh(reg_detector_coords,mesh_elem,mesh_nodes)
    #reg_detector_coords[out_mask] += detector_directions[out_mask] * 0.1

    #plot probe registration
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(mesh_nodes[:,0],mesh_nodes[:,1],mesh_nodes[:,2],s=0.5,alpha=0.05, color='tan')
    ax.scatter(reg_source_coords[:,0],reg_source_coords[:,1],reg_source_coords[:,2], color='red')
    ax.scatter(reg_detector_coords[:,0],reg_detector_coords[:,1],reg_detector_coords[:,2], color='blue')
    
    ax.quiver(reg_source_coords[:,0],reg_source_coords[:,1],reg_source_coords[:,2],
              source_directions[:,0],source_directions[:,1],source_directions[:,2],
              color='red',length=15,)
    
    # ax.quiver(reg_detector_coords[:,0],reg_detector_coords[:,1],reg_detector_coords[:,2],
    #           detector_directions[:,0],detector_directions[:,1],detector_directions[:,2],
    #           color='blue',length=15,)
    

    ax.view_init(elev=45, azim=45)

    return reg_source_coords, reg_detector_coords, source_directions, detector_directions, init_id_src, init_id_det

def points_in_tetrahedron(points, verts):
    """
    points: (n_points, 3)
    verts: (4, 3) array for single tetrahedron vertices
    Returns: (n_points,) bool array
    """
    A, B, C, D = verts
    v0, v1, v2 = B-A, C-A, D-A
    mat = np.column_stack((v0, v1, v2))  # shape (3, 3)
    inv_mat = np.linalg.inv(mat)
    rel_points = (points - A).T  # shape (3, n_points)
    bary_coords = inv_mat @ rel_points  # shape (3, n_points)
    bary_coords = bary_coords.T
    cond = (
        np.all(bary_coords >= 0, axis=1)
        & (np.sum(bary_coords, axis=1) <= 1)
    )
    return cond

def points_in_tetrahedral_mesh(points, tetras, vertices):
    """
    Checks if given points are inside of head mesh
    points: (n_points, 3)
    tetras: (n_tetras, 4) tetra indices into vertices
    vertices: (n_vertices, 3)
    Returns: (n_points,) bool array indicating if each is inside any tetrahedron
    """
    results = np.zeros(len(points), dtype=bool)
    results_idx = np.zeros(len(points)) - 1
    for tet_idx, tet in enumerate(tetras):
        mask = points_in_tetrahedron(points, vertices[tet])
        results = results | mask
        if np.any(results):
            results_idx[np.where(mask)] = tet_idx
    return results, results_idx

def find_optode_directions(optode_coords, mesh_nodes):
    #TODO needs docstring
    #assumes head is roughly spherical and returns ndarray of unit vectors pointing from input point to mesh center
    optode_dirs = []
    midmeshpoint = (mesh_nodes.max(0) + mesh_nodes.min(0)) / 2
    
    for opt_idx in range(optode_coords.shape[0]):
        
        #finding direction of optode placement towards center of head mesh
        current_optode = optode_coords[opt_idx,:]
        optode_dir = midmeshpoint - current_optode
        
        #converting to unit length for mmc
        unit_optode_dir = optode_dir / np.linalg.norm(optode_dir)
        
        #appending to list
        optode_dirs.append(unit_optode_dir)
        
    #converting to numpy array
    optode_dirs = np.array(optode_dirs)
    return optode_dirs

ArrayLike = Union[List, Tuple, "np.ndarray"]