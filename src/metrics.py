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

def mean_map_entropy(pc_map, map_tips=None, KNN_RAD=1):
    MIN_KNN = 5

    map_tree = o3d.geometry.KDTreeFlann(pc_map)
    points = np.asarray(pc_map.points)
    metric = []
    for i in range(points.shape[0]):
        point = points[i]
        [k, idx, _] = map_tree.search_radius_vector_3d(point, KNN_RAD)
        if len(idx) > MIN_KNN:
            cov = np.cov(points[idx].T)
            det = np.linalg.det(2 * np.pi * np.e * cov)
            if det > 0:
                metric.append(0.5 * np.log(det))

    return 0 if len(metric) == 0 else np.mean(metric)


def mean_plane_variance(pc_map, map_tips=None, KNN_RAD=1):
    MIN_KNN = 5

    map_tree = o3d.geometry.KDTreeFlann(pc_map)
    points = np.asarray(pc_map.points)

    metric = []
    for i in range(points.shape[0]):
        point = points[i]
        [k, idx, _] = map_tree.search_radius_vector_3d(point, KNN_RAD)
        if len(idx) > MIN_KNN:
            cov = np.cov(points[idx].T)
            eigenvalues = np.linalg.eig(cov)[0]
            metric.append(min(eigenvalues))

    return 0 if len(metric) == 0 else np.mean(metric)

def rpe(T_gt, T_est):
    seq_len = len(T_gt)
    err = 0
    for i in range(seq_len):
        for j in range(seq_len):
            d_gt = T_gt[i] @ np.linalg.inv(T_gt[j])
            d_est = T_est[i] @ np.linalg.inv(T_est[j])
            dt = d_est[:3, 3] - d_gt[:3, 3]
            err += np.linalg.norm(dt) ** 2

    return err
