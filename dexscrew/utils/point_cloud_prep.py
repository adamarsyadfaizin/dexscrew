# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: Lessons from Learning to Spin "Pens"
# Copyright (c) 2024 All Authors
# Licensed under MIT License
# https://github.com/HaozhiQi/penspin/
# --------------------------------------------------------
# prepare point cloud files when fallback
import numpy as np


def sample_cylinder(h, num_points=100, num_circle_points=15, side_points=70):
    """Sample points on the surface of a unit-radius cylinder of height h."""
    assert num_points == num_circle_points * 2 + side_points
    pcs = np.zeros((num_points, 3))
    # sample points from top and bottom surfaces
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[:num_circle_points, 0] = r * np.cos(theta) * 0.5
    pcs[:num_circle_points, 1] = r * np.sin(theta) * 0.5
    pcs[:num_circle_points, 2] = 0.5 * h
    r = np.sqrt(np.random.random(num_circle_points))
    theta = np.random.random(num_circle_points) * 2 * np.pi
    pcs[num_circle_points:num_circle_points * 2, 0] = r * np.cos(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 1] = r * np.sin(theta) * 0.5
    pcs[num_circle_points:num_circle_points * 2, 2] = -0.5 * h
    # sample points from the side surface
    vec = np.random.random((side_points, 2)) - 0.5
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    vec *= 0.5
    pcs[num_circle_points * 2:, :2] = vec
    pcs[num_circle_points * 2:, 2] = h * (np.random.random(side_points) - 0.5)
    return pcs
