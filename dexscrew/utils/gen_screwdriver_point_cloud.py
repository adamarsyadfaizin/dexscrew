#!/usr/bin/env python3
"""
Generate point cloud npy files for screwdrivers (shaft + handle).

Shaft points are sampled analytically from URDF cylinder parameters, and
handle points are sampled from the associated mesh using `trimesh`.

Example:
python dexscrew/utils/gen_screwdriver_point_cloud.py --input_dir assets/screw/driver --num_points 100 --shaft_ratio 0.2
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import trimesh

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate point cloud NPY files from screwdriver URDFs')
    parser.add_argument('--input_dir', type=str, default='assets/screw/driver', 
                       help='Directory containing screwdriver URDF files')
    parser.add_argument('--num_points', type=int, default=100,
                       help='Total number of points to sample (shaft + handle combined)')
    parser.add_argument('--shaft_ratio', type=float, default=0.2,
                       help='Fraction of points for shaft (rest goes to handle)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible sampling (default: None)')
    return parser.parse_args()

def read_xml(filename: str) -> ET.Element:
    """Parse an XML file and return the root element."""
    tree = ET.parse(filename)
    return tree.getroot()

def sample_cylinder_points(radius: float, length: float, num_points: int) -> np.ndarray:
    """Generate approximately uniform points on a cylinder (two caps + side)."""
    circle_points = num_points // 3
    side_points = num_points - 2 * circle_points

    points = []

    angles = np.random.uniform(0, 2*np.pi, circle_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, circle_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.full(circle_points, length/2)
    points.append(np.column_stack([x, y, z]))

    angles = np.random.uniform(0, 2*np.pi, circle_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, circle_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.full(circle_points, -length/2)
    points.append(np.column_stack([x, y, z]))

    angles = np.random.uniform(0, 2*np.pi, side_points)
    heights = np.random.uniform(-length/2, length/2, side_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = heights
    points.append(np.column_stack([x, y, z]))

    return np.vstack(points)

def _find_link(urdf_root: ET.Element, link_name: str) -> Optional[ET.Element]:
    for link in urdf_root.findall('link'):
        if link.get('name') == link_name:
            return link
    return None

def _find_joint_by_links(urdf_root: ET.Element, parent_name: str, child_name: str) -> Optional[ET.Element]:
    for joint in urdf_root.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is not None and child is not None and parent.get('link') == parent_name and child.get('link') == child_name:
            return joint
    return None

def resolve_mesh_path(mesh_filename: str, urdf_path: str) -> str:
    """Resolve mesh file path using a set of common search locations."""
    urdf_dir = os.path.dirname(urdf_path)
    project_root = Path(__file__).resolve().parents[2]

    if os.path.isabs(mesh_filename) and os.path.exists(mesh_filename):
        return os.path.normpath(mesh_filename)

    candidates = []
    candidates.append(os.path.normpath(os.path.join(urdf_dir, mesh_filename)))
    basename = os.path.basename(mesh_filename)
    candidates.append(os.path.normpath(str(project_root / 'assets' / 'meshes' / basename)))
    candidates.append(os.path.normpath(str(project_root / 'assets' / 'screw' / 'meshes' / basename)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    assets_dir = project_root / 'assets'
    try:
        for path in assets_dir.rglob(basename):
            if path.is_file():
                return os.path.normpath(str(path))
    except Exception:
        pass

    return candidates[0]

def sample_points_from_mesh(mesh_path: str, num_points: int) -> np.ndarray:
    """Sample points from a mesh surface using trimesh."""
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)

def get_shaft_cylinder_params(urdf_root: ET.Element, urdf_filename: str) -> Tuple[float, float]:
    shaft_link = _find_link(urdf_root, 'shaft')
    if shaft_link is None:
        raise ValueError(f"No link named 'shaft' found in {urdf_filename}")
    visual = shaft_link.find('visual')
    if visual is None:
        raise ValueError(f"No <visual> element in link 'shaft' of {urdf_filename}")
    geometry = visual.find('geometry')
    if geometry is None:
        raise ValueError(f"No <geometry> element in link 'shaft' of {urdf_filename}")
    cylinder = geometry.find('cylinder')
    if cylinder is None:
        raise ValueError(f"Link 'shaft' in {urdf_filename} must use <cylinder> geometry")
    return float(cylinder.get('radius')), float(cylinder.get('length'))

def get_handle_mesh_path(urdf_root: ET.Element, urdf_filename: str, urdf_path: str) -> str:
    handle_link = _find_link(urdf_root, 'handle')
    if handle_link is None:
        raise ValueError(f"No link named 'handle' found in {urdf_filename}")
    visual = handle_link.find('visual')
    if visual is None:
        raise ValueError(f"No <visual> element in link 'handle' of {urdf_filename}")
    geometry = visual.find('geometry')
    if geometry is None:
        raise ValueError(f"No <geometry> element in link 'handle' of {urdf_filename}")
    mesh = geometry.find('mesh')
    if mesh is None:
        raise ValueError(f"Link 'handle' in {urdf_filename} must use <mesh> geometry for handle sampling")
    mesh_filename = mesh.get('filename')
    if not mesh_filename:
        raise ValueError(f"Link 'handle' in {urdf_filename} <mesh> missing 'filename'")
    return resolve_mesh_path(mesh_filename, urdf_path)

def get_handle_joint_offset_z(urdf_root: ET.Element, urdf_filename: str) -> float:
    joint = _find_joint_by_links(urdf_root, 'shaft', 'handle')
    if joint is None:
        return 0.02
    origin = joint.find('origin')
    if origin is None:
        return 0.02
    xyz = origin.get('xyz', '0 0 0').split()
    if len(xyz) != 3:
        return 0.02
    try:
        return float(xyz[2])
    except Exception:
        return 0.02

def generate_shaft_points_from_urdf(urdf_root: ET.Element, urdf_filename: str, num_points: int) -> np.ndarray:
    """Generate shaft points by reading cylinder parameters from the URDF."""
    radius, length = get_shaft_cylinder_params(urdf_root, urdf_filename)
    return sample_cylinder_points(radius, length, num_points)

def generate_screwdriver_point_cloud(urdf_path: str, num_points: int = 100, shaft_ratio: float = 0.2) -> np.ndarray:
    """Generate complete screwdriver point cloud: shaft (URDF cylinder) + handle (mesh)."""
    
    # Parse URDF
    urdf_root = read_xml(urdf_path)
    urdf_filename = os.path.basename(urdf_path)
    
    # Split points between shaft and handle
    shaft_points_count = int(num_points * shaft_ratio)
    handle_points_count = num_points - shaft_points_count
    
    # 1. Generate shaft points from URDF cylinder
    shaft_points = generate_shaft_points_from_urdf(urdf_root, urdf_filename, shaft_points_count) if shaft_points_count > 0 else np.zeros((0, 3), dtype=np.float32)

    # 2. Generate handle points from mesh
    handle_mesh_path = get_handle_mesh_path(urdf_root, urdf_filename, urdf_path)
    handle_points = sample_points_from_mesh(handle_mesh_path, handle_points_count) if handle_points_count > 0 else np.zeros((0, 3), dtype=np.float32)

    # 3. Position handle relative to shaft using the shaft->handle joint z-offset
    handle_offset_z = get_handle_joint_offset_z(urdf_root, urdf_filename)
    handle_points[:, 2] += handle_offset_z
    
    # 5. Combine shaft + handle
    screwdriver_points = np.vstack([shaft_points, handle_points])
    
    # 4. Ensure exact number of points
    if len(screwdriver_points) > num_points:
        indices = np.random.choice(len(screwdriver_points), num_points, replace=False)
        screwdriver_points = screwdriver_points[indices]
    elif len(screwdriver_points) < num_points:
        padding_needed = num_points - len(screwdriver_points)
        indices = np.random.choice(len(screwdriver_points), padding_needed, replace=True)
        screwdriver_points = np.vstack([screwdriver_points, screwdriver_points[indices]])
    
    return screwdriver_points.astype(np.float32)

def main() -> None:
    args = arg_parse()
    if args.seed is not None:
        np.random.seed(args.seed)

    urdf_files = list(Path(args.input_dir).glob('*.urdf'))
    
    if not urdf_files:
        print(f"No URDF files found in {args.input_dir}")
        return

    print(
        f"Generating {args.num_points} points for {len(urdf_files)} screwdrivers "
        f"in {args.input_dir} "
        f"({args.shaft_ratio:.1%} shaft / {1-args.shaft_ratio:.1%} handle)"
    )

    for urdf_path in sorted(urdf_files):
        
        try:
            point_cloud = generate_screwdriver_point_cloud(urdf_path, args.num_points, args.shaft_ratio)
            
            npy_path = str(urdf_path).replace('.urdf', '.npy')
            np.save(npy_path, point_cloud)

            pc_mean = np.mean(point_cloud, axis=0)
            x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
            y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
            z_min, z_max = point_cloud[:, 2].min(), point_cloud[:, 2].max()
            print(
                f"{urdf_path.name}: saved {os.path.basename(npy_path)}, "
                f"shape={point_cloud.shape} "
            )
            
        except Exception as e:
            print(f"ERROR processing {urdf_path.name}: {e}")
            continue
    
    print(f"npy files saved in: {args.input_dir}")

if __name__ == '__main__':
    main()
