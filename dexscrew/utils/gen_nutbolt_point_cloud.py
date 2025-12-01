#!/usr/bin/env python3
"""
Generate point cloud npy files for screw assemblies (bolt + nut).

Bolt points are sampled analytically from URDF cylinder parameters, and
nut points are sampled from the associated mesh using `trimesh`.

Example:
python dexscrew/utils/gen_nutbolt_point_cloud.py --input_dir assets/screw/trinut --num_points 100 --bolt_ratio 0.3
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import trimesh

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate point cloud NPY files from screw URDFs')
    parser.add_argument('--input_dir', type=str, default='assets/screw/train', 
                       help='Directory containing screw URDF files')
    parser.add_argument('--num_points', type=int, default=100,
                       help='Total number of points to sample (bolt + nut combined)')
    parser.add_argument('--bolt_ratio', type=float, default=0.3,
                       help='Fraction of points for bolt shaft (rest goes to nut)')
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
    
    # Top circle (z = length/2)
    angles = np.random.uniform(0, 2*np.pi, circle_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, circle_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.full(circle_points, length/2)
    points.append(np.column_stack([x, y, z]))
    
    # Bottom circle (z = -length/2)
    angles = np.random.uniform(0, 2*np.pi, circle_points)
    radii = np.sqrt(np.random.uniform(0, radius**2, circle_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = np.full(circle_points, -length/2)
    points.append(np.column_stack([x, y, z]))
    
    # Side surface
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

    # 1) If absolute, return if exists
    if os.path.isabs(mesh_filename) and os.path.exists(mesh_filename):
        return os.path.normpath(mesh_filename)

    candidates = []
    # 2) URDF-relative resolution
    candidates.append(os.path.normpath(os.path.join(urdf_dir, mesh_filename)))
    # 3) Fallback to assets/meshes/<basename>
    basename = os.path.basename(mesh_filename)
    candidates.append(os.path.normpath(str(project_root / 'assets' / 'meshes' / basename)))
    # 4) Optional: assets/screw/meshes/<basename>
    candidates.append(os.path.normpath(str(project_root / 'assets' / 'screw' / 'meshes' / basename)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # 5) Last resort: shallow search under assets for basename
    assets_dir = project_root / 'assets'
    try:
        for path in assets_dir.rglob(basename):
            if path.is_file():
                return os.path.normpath(str(path))
    except Exception:
        pass

    # If still not found, return URDF-relative default (will trigger FileNotFoundError later)
    return candidates[0]

def sample_points_from_mesh(mesh_path: str, num_points: int) -> np.ndarray:
    """Sample points from a mesh surface using trimesh."""
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Sample points on mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    return points.astype(np.float32)

def get_bolt_cylinder_params(urdf_root: ET.Element, urdf_filename: str) -> Tuple[float, float]:
    bolt_link = _find_link(urdf_root, 'bolt')
    if bolt_link is None:
        raise ValueError(f"No link named 'bolt' found in {urdf_filename}")
    visual = bolt_link.find('visual')
    if visual is None:
        raise ValueError(f"No <visual> element in link 'bolt' of {urdf_filename}")
    geometry = visual.find('geometry')
    if geometry is None:
        raise ValueError(f"No <geometry> element in link 'bolt' of {urdf_filename}")
    cylinder = geometry.find('cylinder')
    if cylinder is None:
        raise ValueError(f"Link 'bolt' in {urdf_filename} must use <cylinder> geometry")
    return float(cylinder.get('radius')), float(cylinder.get('length'))

def get_nut_mesh_path(urdf_root: ET.Element, urdf_filename: str, urdf_path: str) -> str:
    nut_link = _find_link(urdf_root, 'nut')
    if nut_link is None:
        raise ValueError(f"No link named 'nut' found in {urdf_filename}")
    visual = nut_link.find('visual')
    if visual is None:
        raise ValueError(f"No <visual> element in link 'nut' of {urdf_filename}")
    geometry = visual.find('geometry')
    if geometry is None:
        raise ValueError(f"No <geometry> element in link 'nut' of {urdf_filename}")
    mesh = geometry.find('mesh')
    if mesh is None:
        raise ValueError(f"Link 'nut' in {urdf_filename} must use <mesh> geometry for nut sampling")
    mesh_filename = mesh.get('filename')
    if not mesh_filename:
        raise ValueError(f"Link 'nut' in {urdf_filename} <mesh> missing 'filename'")
    return resolve_mesh_path(mesh_filename, urdf_path)

def get_nut_joint_offset_z(urdf_root: ET.Element, urdf_filename: str) -> float:
    joint = _find_joint_by_links(urdf_root, 'bolt', 'nut')
    if joint is None:
        # Default to 0.1 to maintain backward-compatible behavior if joint not found
        return 0.1
    origin = joint.find('origin')
    if origin is None:
        return 0.1
    xyz = origin.get('xyz', '0 0 0').split()
    if len(xyz) != 3:
        return 0.1
    try:
        return float(xyz[2])
    except Exception:
        return 0.1

def generate_bolt_points_from_urdf(urdf_root: ET.Element, urdf_filename: str, num_points: int) -> np.ndarray:
    """Generate bolt shaft points by reading cylinder parameters from the URDF."""
    radius, length = get_bolt_cylinder_params(urdf_root, urdf_filename)
    return sample_cylinder_points(radius, length, num_points)

def generate_screw_point_cloud(urdf_path: str, num_points: int = 100, bolt_ratio: float = 0.3) -> np.ndarray:
    """Generate complete screw point cloud: bolt (URDF cylinder) + nut (mesh)."""
    urdf_root = read_xml(urdf_path)
    urdf_filename = os.path.basename(urdf_path)
    
    bolt_points_count = int(num_points * bolt_ratio)
    nut_points_count = num_points - bolt_points_count
    
    bolt_points = generate_bolt_points_from_urdf(urdf_root, urdf_filename, bolt_points_count) if bolt_points_count > 0 else np.zeros((0, 3), dtype=np.float32)

    nut_mesh_path = get_nut_mesh_path(urdf_root, urdf_filename, urdf_path)
    nut_points = sample_points_from_mesh(nut_mesh_path, nut_points_count) if nut_points_count > 0 else np.zeros((0, 3), dtype=np.float32)
    
    nut_offset_z = get_nut_joint_offset_z(urdf_root, urdf_filename)
    nut_points[:, 2] += nut_offset_z
    
    screw_points = np.vstack([bolt_points, nut_points])
    
    if len(screw_points) > num_points:
        indices = np.random.choice(len(screw_points), num_points, replace=False)
        screw_points = screw_points[indices]
    elif len(screw_points) < num_points:
        padding_needed = num_points - len(screw_points)
        indices = np.random.choice(len(screw_points), padding_needed, replace=True)
        screw_points = np.vstack([screw_points, screw_points[indices]])
    
    return screw_points.astype(np.float32)


def main() -> None:
    args = arg_parse()
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Find all URDF files in input directory
    urdf_files = list(Path(args.input_dir).glob('*.urdf'))
    
    if not urdf_files:
        print(f"No URDF files found in {args.input_dir}")
        return

    print(
        f"Generating {args.num_points} points for {len(urdf_files)} screws "
        f"in {args.input_dir} "
        f"({args.bolt_ratio:.1%} bolt / {1-args.bolt_ratio:.1%} nut)"
    )

    for urdf_path in sorted(urdf_files):
        
        try:
            point_cloud = generate_screw_point_cloud(urdf_path, args.num_points, args.bolt_ratio)
            
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
