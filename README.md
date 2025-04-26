<div align="center">

# ⚡️Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass


${{\color{Red}\Huge{\textsf{  CVPR\ 2025\ \}}}}\$


[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2501.13928)
[![Project Website](https://img.shields.io/badge/Fast3R-Website-4CAF50?logo=googlechrome&logoColor=white)](https://fast3r-3d.github.io/)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://fast3r.ngrok.app/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512/)
</div>

![Teaser Image](assets/teaser.png)

Official implementation of **Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass**, CVPR 2025

*[Jianing Yang](https://jedyang.com/), [Alexander Sax](https://alexsax.github.io/), [Kevin J. Liang](https://kevinjliang.github.io/), [Mikael Henaff](https://www.mikaelhenaff.net/), [Hao Tang](https://tanghaotommy.github.io/), [Ang Cao](https://caoang327.github.io/), [Joyce Chai](https://web.eecs.umich.edu/~chaijy/), [Franziska Meier](https://fmeier.github.io/), [Matt Feiszli](https://www.linkedin.com/in/matt-feiszli-76b34b/)*

## ⚠️ Prerequisites ⚠️  
Before running **Fast3R for Windows** please make sure the *base* software stack is already in place.

| What you need | Version | Where to get it | Notes |
|---------------|---------|-----------------|-------|
| **CUDA Toolkit** | 12.4 | [NVIDIA Downloads](https://developer.nvidia.com/cuda-downloads) | The installer adds *nvcc* & device libraries to *`%CUDA_PATH%`*. |
| **GPU driver** | 550.xx or newer | GeForce / RTX Studio driver page | Must match the CUDA runtime shipped with PyTorch-CUDA 12.4. |
| **Visual Studio Build Tools** | 2022 | [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | Needed only once to compile PyTorch3D (≈ 10 min). |
| **git** | any recent | [git-scm.com](https://git-scm.com/) | For cloning & pulling sub-modules. |

> **Tip (UTF-8 console)** — open *PowerShell* and run **`chcp 65001`** once:  
> it switches the current window to UTF-8 so progress bars / emojis show up correctly.

## Installation

```powershell
# clone project
git clone https://github.com/RonnieyL/Fast3rWindows
cd Fast3rWindows

# create conda env
conda create -n fast3r python=3.11 cmake=3.14.0 -y
conda activate fast3r

# PyTorch (+CUDA 12.4 runtime)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia

# PyTorch3D (compiled from source)  ⚠️ takes a while
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# project requirements
pip install -r requirements.txt

# install Fast3R package in-place
pip install -e .
```

### Why doesn't the demo crash on Windows now?  
We apply three tiny patches automatically when you import the repo:

1. **UTF-8 everywhere** – at the top of `fast3r/viz/demo.py` we add  
   ```python
   if sys.platform == "win32":
       sys.stdin.reconfigure(encoding="utf-8")
       sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
       sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
   ```
2. **ViserServerManager** – re-implemented with a *static* `run_worker` method so
   the multiprocessing target is pickle-safe on Windows (prevents
   `TypeError: cannot pickle '_thread.RLock' object`).
3. **No `cuROPE`** – the README explicitly says *do not* install the DUSt3R
   `cuROPE` extension; Fast3R already ships a pure-PyTorch fallback.

## Demo

Simply run:
```powershell
chcp 65001 
set PYTHONUTF8=1
python fast3r/viz/demo.py
```

This will start the Gradio interface where you can upload images or video for reconstruction.

## Command-line Utilities

### `convert.py` — Batch Processing to COLMAP Format

Process multiple images and export in various formats:
```powershell
python convert.py --image_dir ./data/keyboard/images ^
                  --output_dir ./outputs/keyboard ^
                  --image_size 512
```

Outputs:
- `scene.ply` - Colored global point cloud
- `poses_c2w.npy` - 4×4 extrinsics matrices
- `intrinsics.npz` - Focal lengths and dimensions per view
- COLMAP format files (`cameras.txt`, `images.txt`, `points3D.txt`)

**Note:** Use `--image_size 224` when processing >20 images on GPUs with ≤8GB VRAM.

### `view_pc.py` — Quick Point Cloud Viewer

For fast visualization without starting the full viser server:
```powershell
python view_pc.py outputs/keyboard/scene.ply
```

## Using Fast3R in Your Own Project

To use Fast3R in your own project, you can import the `Fast3R` class from `fast3r.models.fast3r` and use it as a regular PyTorch model.

```python
import torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

# --- Setup ---
# Load the model from Hugging Face
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

# Set model to evaluation mode
model.eval()
lit_module.eval()

# --- Load Images ---
# Provide a list of image file paths. Images can come from different cameras and aspect ratios.
filelist = ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
images = load_images(filelist, size=512, verbose=True)

# --- Run Inference ---
# The inference function returns a dictionary with predictions and view information.
output_dict, profiling_info = inference(
    images,
    model,
    device,
    dtype=torch.float32,  # or use torch.bfloat16 if supported
    verbose=True,
    profiling=True,
)

# --- Estimate Camera Poses ---
# This step estimates the camera-to-world (c2w) poses for each view using PnP.
poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
    output_dict['preds'],
    niter_PnP=100,
    focal_length_estimation_method='first_view_from_global_head'
)
# poses_c2w_batch is a list; the first element contains the estimated poses for each view.
camera_poses = poses_c2w_batch[0]

# Print camera poses for all views.
for view_idx, pose in enumerate(camera_poses):
    print(f"Camera Pose for view {view_idx}:")
    print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

# --- Extract 3D Point Clouds for Each View ---
# Each element in output_dict['preds'] corresponds to a view's point map.
for view_idx, pred in enumerate(output_dict['preds']):
    point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
    print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  # shape: (1, 368, 512, 3), i.e., (1, Height, Width, XYZ)
```


## Known Issues & Workarounds

These warnings are benign and can be safely ignored:

| Message | What it means | Impact |
|---------|---------------|--------|
| `Warning, cannot find cuda-compiled version of RoPE2D` | Using pure-PyTorch Rotary PE (≈3ms slower per image) | ❌ None |
| `pl_bolts UnderReviewWarning ...` | Lightning-Bolts internal warning | ❌ None |
| `encode_images time: ... something is wrong with the encoder` | Debug print (ignore unless time >1s per view) | ❌ None |

## Future Implementations

- **Enhanced CLI for `convert.py`**
  - Flexible output format selection (PLY/NPY/COLMAP)
  - Optional depth/confidence map exports
- **Standalone Viser Viewer**
  - Load existing reconstructions without re-running inference
- **Windows-optimized Distribution**
  - Pre-built PyTorch3D wheels
  - Automatic VRAM management
  - Batch GIF renderer for sequences

## Dataset Preprocessing

Please follow [DUSt3R's data preprocessing instructions](https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#datasets) to prepare the data for training and evaluation. The pre-processed data is compatible with the [multi-view dataloaders](fast3r/dust3r/datasets) in this repo.

For preprocessing the DTU, 7-Scene, and NRGBD datasets for evaluation, we follow [Spann3r's data processing instructions](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md).

## FAQ

- Q: `httpcore.ConnectError: All connection attempts failed` when launching the demo?
  - See [#34](https://github.com/facebookresearch/fast3r/issues/34). Download the example videos into a local directory.
- Q: Data pre-processing for BlendedMVS, `train_list.txt` is missing?
  - See [#33](https://github.com/facebookresearch/fast3r/issues/33).
- Q: Loading checkpoint to fine-tune Fast3R?
  - See [#25](https://github.com/facebookresearch/fast3r/issues/25)
- Q: Running demo on Windows? (TypeError: cannot pickle '_thread.RLock' object)
  - See [#28](https://github.com/facebookresearch/fast3r/issues/28). It seems that some more work is needed to make the demo compatible with Windows - we hope the community could contribute a PR!
- Q: Completely messed-up point cloud output?
  - See [#21](https://github.com/facebookresearch/fast3r/issues/21). Please make sure the cuROPE module is NOT installed.
- Q: My GPU doesn't support FlashAttention / `No available kernel. Aborting execution`?
  - See [#17](https://github.com/facebookresearch/fast3r/issues/17). Use `attn_implementation=pytorch_auto` option instead.
- Q: `TypeError: Fast3R.__init__() missing 3 required positional arguments: 'encoder_args', 'decoder_args', and 'head_args'`
  - See See [#7](https://github.com/facebookresearch/fast3r/issues/7). It is caused by a networking issue with downloading the model from Huggingface in some countries (e.g., China) - please pre-download the model checkpoint with a working networking configuration, and use a local path to load the model instead.
## License

The code and models are licensed under the [FAIR NC Research License](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Citation

```
@InProceedings{Yang_2025_Fast3R,
    title={Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass},
    author={Jianing Yang and Alexander Sax and Kevin J. Liang and Mikael Henaff and Hao Tang and Ang Cao and Joyce Chai and Franziska Meier and Matt Feiszli},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2025},
}
```

## Acknowledgement

Fast3R is built upon a foundation of remarkable open-source projects. We deeply appreciate the contributions of these projects and their communities, whose efforts have significantly advanced the field and made this work possible.

- [DUSt3R](https://dust3r.europe.naverlabs.com/)
- [Spann3R](https://hengyiwang.github.io/projects/spanner)
- [Viser](https://viser.studio/main/)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)
