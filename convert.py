#!/usr/bin/env python3
"""
Fast-3-R → COLMAP  (plus handy PLY)

After it finishes you will have:

    <output_dir>/
        sparse_0/0/          # cameras.bin / images.bin / points3D.bin
        reconstruction.ply   # coloured point cloud for a quick look
"""

import os
import torch
import numpy as np
from pathlib import Path
import argparse

from fast3r.dust3r.inference_multiview import inference
from fast3r.utils.checkpoint_utils import load_model
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from utils.sfm_utils import init_filestructure, load_images
from scene.colmap_loader import (
    Camera as ColmapCamera,
    BaseImage as ColmapImage,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
)
from scene.colmap_loader import Point3D, write_points3D_binary

#─────────────────────────────────────────────────────────────────────────────
def save_ply(path: Path, points, colors):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    data = np.empty(len(points), dtype=[
        ("xyz", np.float32, 3),
        ("rgb", np.uint8, 3),
    ])
    data["xyz"] = points
    data["rgb"] = colors
    with open(path, 'wb') as f:
        f.write(("\n".join(header) + "\n").encode('ascii'))
        f.write(data.tobytes())

def save_points3dbin(bin_path: Path, points: np.ndarray, colors: np.ndarray, error: float = 0.0):
    # build a dict of Point3D entries
    pts3D = {}
    for i in range(len(points)):
        pts3D[i] = Point3D(
            id=i,
            xyz=points[i].astype(np.float64),
            rgb=colors[i].astype(np.uint8),
            error=error,
            image_ids=np.empty((0,), dtype=np.int64),
            point2D_idxs=np.empty((0,), dtype=np.int64),
        )
    write_points3D_binary(pts3D, str(bin_path))
    print(f"Wrote {len(points):,} points to {bin_path}")
    return len(points)

def save_cameras_bin_txt(out_dir: Path, intrinsics: np.ndarray, width: int, height: int):
    cams = {}
    # intrinsics is shape (N,3,3)
    for i, K in enumerate(intrinsics, start=1):
        fx, fy = float(K[0,0]), float(K[1,1])
        cx, cy = float(K[0,2]), float(K[1,2])
        cams[i] = ColmapCamera(
            id=i,
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy],
        )
    write_cameras_binary(cams, out_dir / "cameras.bin")
    write_cameras_text(cams, out_dir / "cameras.txt")
    print(f"Wrote cameras.bin / .txt to {out_dir}")

def save_images_bin_txt(out_dir: Path, extrinsics_w2c: np.ndarray, image_files: list[str]):
    imgs = {}
    from scene.colmap_loader import rotmat2qvec
    for i, (w2c, path) in enumerate(zip(extrinsics_w2c, image_files), start=1):
        R = w2c[:3, :3]
        t = w2c[:3,  3]
        qvec = rotmat2qvec(R)
        imgs[i] = ColmapImage(
            id=i,
            qvec=qvec,
            tvec=t,
            camera_id=i,
            name=Path(path).name,
            xys=[],
            point3D_ids=[]
        )
    write_images_binary(imgs, out_dir / "images.bin")
    write_images_text(imgs, out_dir / "images.txt")
    print(f"Wrote images.bin / .txt to {out_dir}")

#─────────────────────────────────────────────────────────────────────────────
def process_directory(
    image_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    image_size: int = 512,
    keep_percent: float = 0.1,
):
    assert 0 < keep_percent <= 1
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path, sparse_0_path, _ = init_filestructure(output_dir)

    # load & resize
    print(f">> Loading images from {image_dir}")
    imgs, (orgW, orgH) = load_images(sorted([str(p) for p in image_dir.iterdir()]), size=image_size)
    image_files = [str(p) for p in image_dir.iterdir() if p.suffix.lower() in (".jpg",".png")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, lit_module = load_model(checkpoint_dir, device=device)

    # inference
    print(">> Running inference...")
    output_dict, _ = inference(imgs, model, device, dtype=torch.float32, verbose=True, profiling=True)

    print(">> Aligning global points...")
    lit_module.align_local_pts3d_to_global(output_dict["preds"], output_dict["views"], min_conf_thr_percentile=85)

    # global alignment, poses
    print(">> Estimating camera poses...")
    poses_c2w_batch, focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"], niter_PnP=100, focal_length_estimation_method="first_view_from_global_head"
    )
    poses_c2w = poses_c2w_batch[0]
    extrinsics_w2c = np.stack([np.linalg.inv(p) for p in poses_c2w])

    for group_name in ("preds", "views"):
        for d in output_dict[group_name]:
            for key, val in list(d.items()):
                if isinstance(val, torch.Tensor):
                    d[key] = val.cpu().numpy()

    # collect all points + colours
    pts = []
    cols= []
    for pred, view in zip(output_dict["preds"], output_dict["views"]):
        p = pred["pts3d_in_other_view"].reshape(-1,3)
        imgs_rgb = ((view["img"].squeeze().transpose(1,2,0)+1)*127.5).astype(np.uint8)
        c = imgs_rgb.reshape(-1,3)
        pts.append(p); cols.append(c)
    pts3d  = np.concatenate(pts, axis=0)
    colors = np.concatenate(cols,axis=0)

    # sample
    N = len(pts3d)
    idx = np.random.choice(N, size=int(N*keep_percent), replace=False)
    pts3d, colors = pts3d[idx], colors[idx]
    print(f">>Ranomly sampled {len(pts3d):,} points from {N:,} total")

    # your intrinsics matrices at the **resized** resolution
    # build them explicitly rather than scaling
    H, W = output_dict["views"][0]["img"].shape[2:]
    Ks = []
    for f in focals[0]:
        Ks.append(np.array([[f,0,W/2],
                            [0,f,H/2],
                            [0,0,  1]], dtype=np.float32))
    Ks = np.stack(Ks, 0)

    print("→ Writing COLMAP files...")
    print(f">> Saving intrinsics")
    save_cameras_bin_txt (sparse_0_path, Ks, W, H)
    print(f">> Saving extrinsics")
    save_images_bin_txt  (sparse_0_path, extrinsics_w2c, image_files)
    print(f">> Saving points3D")
    save_points3dbin     (sparse_0_path / "points3D.bin", pts3d, colors)
    print(f">> Saving PLY")
    save_ply             (output_dir / "reconstruction.ply", pts3d, colors)

    return sparse_0_path

# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",  required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--checkpoint_dir", type=Path, default=".")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--keep_percent", type=float, default=0.1)
    args = p.parse_args()
    out = process_directory(
        args.image_dir,
        args.output_dir,
        args.checkpoint_dir,
        args.image_size,
        args.keep_percent
    )
    print("done →", out)

if __name__=="__main__":
    main()
