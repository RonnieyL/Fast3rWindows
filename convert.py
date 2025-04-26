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
import collections
from pathlib import Path
from tqdm import tqdm
import argparse

from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
from fast3r.utils.checkpoint_utils import load_model
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from utils.sfm_utils import (
    save_intrinsics, save_extrinsic, save_points3D, init_filestructure,
    Camera, BaseImage, Point3D, compute_co_vis_masks
)

#──────────────────────────────Helper save ply func────────────────────────────────

def save_ply(path: Path, points, colors):
    """Save point cloud to PLY file."""
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
    
    # Create structured array
    data = np.empty(len(points), dtype=[
        ("xyz", np.float32, 3),
        ("rgb", np.uint8, 3),
    ])
    data["xyz"] = points
    data["rgb"] = colors

    # Write PLY file
    with open(path, 'wb') as f:
        f.write(("\n".join(header) + "\n").encode('ascii'))
        f.write(data.tobytes())

# ─────────────────────────────────────────────────────────────────────────────


def process_directory(
    image_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    image_size: int = 512,
):
    """Run Fast-3-R, export COLMAP & PLY."""

    # ------------------------------------------------------------------ setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path, sparse_0_path, _ = init_filestructure(output_dir)

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(
        [f for f in Path(image_dir).glob("*") if f.suffix.lower() in valid_ext]
    )
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    image_files_str = [str(f) for f in image_files]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, lit_module = load_model(checkpoint_dir, device=device)

    # ---------------------------------------------------------- load & resize
    imgs = load_images(
        image_files_str,
        size=image_size,
        verbose=True,
    )

    # ---------------------------------------------------------------- infer
    print(">> Inference …")
    output_dict, _ = inference(
        imgs,
        model,
        device,
        dtype=torch.float32,
        verbose=True,
        profiling=True,
    )

    # tensors → cpu numpy
    for pred in output_dict["preds"]:
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred[k] = v.cpu()
    for view in output_dict["views"]:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.cpu()

    # ---------------------------------------------------- global registration
    print(">> Global alignment …")
    lit_module.align_local_pts3d_to_global(
        preds=output_dict["preds"],
        views=output_dict["views"],
        min_conf_thr_percentile=85,
    )

    poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
        output_dict["preds"],
        niter_PnP=100,
        focal_length_estimation_method="first_view_from_global_head",
    )
    poses_c2w = poses_c2w_batch[0]

    def to_numpy(m):
        """Torch tensor → np.ndarray, otherwise return unchanged."""
        return m.detach().cpu().numpy() if hasattr(m, "detach") else m

    extrinsics_w2c = np.array([np.linalg.inv(to_numpy(p)) for p in poses_c2w])

    # --------------------------------------------------------- gather points
    pts3d_list, col_list, conf_list = [], [], []
    for pred, view in zip(output_dict["preds"], output_dict["views"]):
        pts = pred["pts3d_in_other_view"].cpu().numpy().reshape(-1, 3)
        conf = pred["conf"].cpu().numpy().flatten()

        img_rgb = view["img"].cpu().squeeze().permute(1, 2, 0).numpy()  # [-1,1]
        colors = ((img_rgb.reshape(-1, 3) + 1) * 127.5).astype(np.uint8)

        pts3d_list.append(pts)
        conf_list.append(conf)
        col_list.append(colors)

    pts3d = np.concatenate(pts3d_list, axis=0)
    colors = np.concatenate(col_list, axis=0)
    confs  = np.concatenate(conf_list, axis=0)

    # -------------------------------------------------------------- save all
    print(">> Saving COLMAP model …")

    focals = estimated_focals[0]                #  shape [n_views]
    height, width = output_dict["views"][0]["img"].shape[2:4]
    org_H, org_W = output_dict["views"][0]["true_shape"][0]

    intrinsics = []
    for f in focals:
        K = np.array([[f, 0,  width/2],
                    [0, f,  height/2],
                    [0, 0,        1 ]], dtype=np.float32)
        intrinsics.append(K)
    intrinsics = np.stack(intrinsics)

    def pts_to_depth(pts_cam, h, w):
        """
        pts_cam : (N,3) points already expressed in *camera* coordinates
        returns  : (h,w) depth map with np.inf where no point projects
        """
        z = pts_cam[:, 2]                              # camera-space depth (positive)
        # Project to pixel coordinates ----------------------------
        xy_h = intrinsics[i] @ pts_cam.T               # (3,N)   K·[x y z]^T
        xy    = (xy_h[:2] / xy_h[2]).T                 # (N,2)   divide by z
        xy    = np.round(xy).astype(int)

        depth = np.full((h, w), np.inf, np.float32)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & \
            (xy[:, 1] >= 0) & (xy[:, 1] < h) & (z > 0)

        # z-buffer: keep closest point per pixel
        for px, py, zz in zip(xy[ok, 0], xy[ok, 1], z[ok]):
            if zz < depth[py, px]:
                depth[py, px] = zz
        return depth

    h, w = height, width
    depthmaps = []
    for i in range(len(output_dict["preds"])):               # one per view
        # transform global points to current camera space -----------------
        R = extrinsics_w2c[i, :3, :3]
        t = extrinsics_w2c[i, :3,  3]
        pts_cam = (R @ pts3d.T).T + t                        # (N,3)

        depthmaps.append(pts_to_depth(pts_cam, h, w))

    depthmaps = np.stack(depthmaps)      # shape (n_views, H, W)

    avg_conf = confs.reshape(len(output_dict["preds"]), -1).mean(1)
    sorted_conf_indices = np.argsort(avg_conf)[::-1]

    image_sizes = (len(output_dict["preds"]), h, w, 3)   # last dim dummy=3
    overlapping_masks = compute_co_vis_masks(
        sorted_conf_indices,
        depthmaps,
        pts3d,
        intrinsics,
        extrinsics_w2c,
        image_sizes,
        depth_threshold=0.01      # tweak if necessary
    )

    # intrinsics
    print(">> Saving intrinsics …")
    save_intrinsics(
        sparse_0_path,
        focals,
        [org_W, org_H],
        [height, width, 3],
        save_focals=True,
    )

    # extrinsics
    print(">> Saving extrinsics …")
    image_suffix = image_files[0].suffix
    save_extrinsic(
        sparse_0_path,
        extrinsics_w2c,
        [f.stem for f in image_files],
        image_suffix,
    )

    # points
    print(">> Saving 3d points …")
    imgs_np = np.stack([v["img"].cpu().squeeze().permute(1, 2, 0).numpy() for v in output_dict["views"]])
    imgs_np = np.clip(imgs_np, -1.0, 1.0)

    pts_num = save_points3D(
        sparse_0_path,
        imgs_np,
        pts3d,
        confs,
        overlapping_masks,
        use_masks=False,
        save_all_pts=True,
        save_txt_path=output_dir,   # or model_path
        depth_threshold=0.01
    )


    print(f"   ↳ cameras / images / points written (pts = {pts_num:,})")
    save_ply(output_dir / "reconstruction.ply", pts3d, colors)
    print(f"   ↳ PLY preview saved in: {output_dir / 'reconstruction.ply'}")

    return sparse_0_path


# ═════════════════════════════════════════════════════════════════════════════
#                                    CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser("Fast-3-R to COLMAP exporter")
    p.add_argument("--image_dir",   required=True, type=Path,
                   help="folder with input images")
    p.add_argument("--output_dir",  required=True, type=Path,
                   help="destination folder")
    p.add_argument("--checkpoint_dir", type=Path,
                   default=Path("--checkpoint_dir"),
                   help="Fast-3-R checkpoint directory")
    p.add_argument("--image_size", type=int, default=512,
                   help="resize long side to this (default 512)")
    args = p.parse_args()

    sparse_path = process_directory(
        args.image_dir,
        args.output_dir,
        args.checkpoint_dir,
        args.image_size,
    )

    print("\nReconstruction complete!")
    print(f"COLMAP model in: {sparse_path}")
    print(f"PLY preview:     {args.output_dir / 'reconstruction.ply'}")


if __name__ == "__main__":
    main()
