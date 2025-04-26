#!/usr/bin/env python
"""
Quick viewer for the point-clouds saved by fast3rstore.py
--------------------------------------------------------

 • Pass a *.ply* path  → colours are read from the file  ✅ perfect colour
 • Pass a *.npy* path  → file is assumed to contain only XYZ; the viewer
                         sets every point to mid-gray so you can at least
                         inspect the geometry                                            

Usage
~~~~~
    python view_cloud.py path/to/scene.ply
    python view_cloud.py path/to/view_00_pointcloud.npy
"""
import sys, os, numpy as np
import open3d as o3d

def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)          # colours already inside
        if len(pcd.colors) == 0:                     # just in case
            print("[viewer] warning: PLY had no colours, painting gray")
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        return pcd

    if ext == ".npy":
        pts = np.load(path)                          # (H,W,3) or (N,3)
        pts = pts.reshape(-1, 3).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])     # gray fallback
        return pcd

    raise RuntimeError(f"Unknown file type: {ext}")

def main():
    if len(sys.argv) != 2:
        exe = os.path.basename(sys.argv[0])
        print(f"usage:  {exe}  path/to/scene.ply | view_X_pointcloud.npy")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        sys.exit(f"[viewer] file not found: {path}")

    pcd = load_pointcloud(path)
    o3d.visualization.draw_geometries(
        [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
        window_name=f"Fast3R viewer – {os.path.basename(path)}"
    )

if __name__ == "__main__":
    main()