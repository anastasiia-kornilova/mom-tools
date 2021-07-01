#  Copyright 2021 Anastasiia Kornilova
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
import open3d as o3d
from random_geometry_points.plane import Plane
import scipy


def generate_planes(N_poses):
    R = 1
    ref_point = (0.0, 0.0, 0.0)
    max_plane_dist = 5
    v1 = (1.0, 0.0, 0.0)
    v2 = (0.0, 1.0, 0.0)
    v3 = (0.0, 0.0, 1.0)
    
    plane_normals = [v1, v2, v3]
    planes = []
    transforms = []

    N_max_planes = 7
    # Uniformly sample number of additional points
    N_planes = np.random.randint(0, N_max_planes + 1)

    # Generate plane normal uniformly on R3 sphere
    for i in range(N_planes):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        v = (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
        plane_normals.append(v)

    # Create planes from plane normals
    for i in range(N_planes + 3):
        p = Plane(plane_normals[i], 0.0, ref_point, R)
        planes.append(p)

    # Uniformly sample plane shift with respect to pc
    i = False
    transforms = []
    while i == False:
        transforms = []
        for i in range(N_planes + 3):
            x = np.random.uniform(-max_plane_dist, max_plane_dist)
            y = np.random.uniform(-max_plane_dist, max_plane_dist)
            z = np.random.uniform(-max_plane_dist, max_plane_dist)
            transforms.append((x, y, z))
    
        paired_dist = scipy.spatial.distance.cdist(transforms, transforms)
        i = len(np.where((paired_dist < 3) & (paired_dist > 0))[0]) == 0 

    densities = []
    for i in range(N_planes + 3):
        densities.append(np.random.randint(10, 100))

    pcs_in_map = []
    orth_subset = []
    for pose_id in range(N_poses):
        pose_pc = o3d.geometry.PointCloud()
        for i in range(N_planes + 3):
            plane_points = planes[i].create_random_points(densities[i])
            plane_pc = o3d.geometry.PointCloud()
            plane_pc.points = o3d.utility.Vector3dVector(plane_points)
            plane_pc.translate(transforms[i])
            
            if i < 3 and pose_id == 0:
                orth_subset.append(copy.deepcopy(np.asarray(plane_pc.points)))

            pose_pc += plane_pc

        pcs_in_map.append(pose_pc)
        
    return pcs_in_map, orth_subset
