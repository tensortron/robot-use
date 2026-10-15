# Environment

## Platform (as measured)

| | |
|---|---|
| Board | Jetson Orin (`nvidia-smi` reports device name `Orin`) |
| L4T / JetPack | R39.2 (JetPack 7) |
| OS | Ubuntu 24.04.4 LTS, kernel `6.8.12-1021-tegra`, aarch64 |
| RAM | 61 GB, **no swap** |
| GPU driver | 595.78, CUDA 13.2 |
| GPU compute capability | **8.7** |
| CUDA toolkit | **not installed** — no `nvcc`, no `/usr/local/cuda`. Not needed; pip wheels ship their own runtime. |
| Python | 3.12.3 system, `uv` at `~/.local/bin/uv` |
| Power mode | MAXN (`nvpmodel -m 0`, `jetson_clocks`) |

## The environment

`<repo>/.venv`, Python 3.12, with `lerobot` installed **editable** from
`~/code/lerobot` (branch `main`).

Verified working set:

```
torch        2.11.0+cu130
torchvision  0.26.0+cu130
torchcodec   0.11.1+cu130
numpy        2.2.6
opencv       4.13.0 (opencv-python-headless)
```

## Why torch 2.11 exactly — not 2.12/2.13/2.14

The version is fully determined by lerobot's dependency graph on aarch64. It is
not a conservative choice; it is the only point in the solution space:

- `lerobot` pins `torch>=2.7,<2.12.0` (`pyproject.toml:62`)
- on linux aarch64 it pins `torchcodec>=0.11.0,<0.12.0` (`pyproject.toml:113`)
- torchcodec 0.11 requires `torch>=2.11`; torchcodec 0.12 requires `torch==2.12`

Intersecting those leaves **torch 2.11.x and nothing else**.

### Why not the CUDA 13.2 (`cu132`) wheels

A `cu132` index does exist at `https://download.pytorch.org/whl/cu132`, and the
latest torch is 2.14.0. Two independent reasons we don't use it:

1. **The cu132 index's oldest torch is 2.12.0**, which is exactly the version
   lerobot excludes (`<2.12.0`). The ranges are disjoint. Using it would mean
   relaxing *two* pins (`torch<2.12` → `<2.13` and `torchcodec<0.12` → `<0.13`),
   putting the rig outside lerobot's tested dependency set.
2. **CUDA minor versions are forward-compatible.** `+cu130` describes the
   toolkit the wheel was *built with*, not a requirement on the driver. A cu130
   build runs fine on a 13.2 driver — verified on this box, not assumed:
   `torch.cuda.is_available() == True`, GPU matmul executed, device `Orin (8, 7)`.

So cu132 would buy newer CUDA libraries, not compatibility, at the cost of
leaving lerobot's supported range.

## The `sm_87` warning (harmless — expected on every run)

```
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
```

The aarch64 wheel's arch list is `['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']`
— it is the generic SBSA server build and does not include Orin's `sm_87`. It
still works: sm_80 cubins run on sm_87 via CUDA minor-version binary
compatibility. Kernels are simply not Orin-tuned.

This is a build-time `TORCH_CUDA_ARCH_LIST` choice, **not** a function of the
CUDA version — moving to cu132 would not add `sm_87`. The only way to get truly
Orin-tuned kernels is building torch from source with `TORCH_CUDA_ARCH_LIST=8.7`
(multi-hour). Not worth it here: at 25 fps with three cameras the bottleneck is
the USB bus and Feetech serial buses, not matmuls.

`pypi.jetson-ai-lab.io` has **no JetPack 7 index** (only `jp6/cu126|cu128|cu129`
and `sbsa/cu130`), so PyPI is the correct source.

## Install recipe (reproducible from scratch)

System prerequisites (need root):

```bash
sudo usermod -aG dialout $USER      # REQUIRED for /dev/ttyACM*; log out/in after
sudo apt install -y ffmpeg          # REQUIRED by torchcodec
sudo nvpmodel -m 0 && sudo jetson_clocks
```

Then (by hand, not `uv sync`: `pyproject.toml` and `uv.lock` serve other machines, and would replace the CUDA
torch with PyPI's):

```bash
cd <repo>
uv venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install -e "$HOME/code/lerobot[feetech,core_scripts]"

# Fix 1 + 2 (see below) — MUST come after, and MUST use --index-url
VIRTUAL_ENV=$PWD/.venv uv pip install --reinstall --index-url https://pypi.org/simple \
    "torch==2.11.0" "torchvision==0.26.0"
VIRTUAL_ENV=$PWD/.venv uv pip install --index-url https://pypi.org/simple \
    "nvidia-npp" "numpy<2.3"
```

Verify:

```bash
./.venv/bin/python -c "
import torch, torchvision, torchcodec, numpy, cv2
from torchvision.ops import nms
from torchcodec.decoders import VideoDecoder
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.teleoperators.bi_so_leader import BiSOLeader
print(torch.__version__, torchvision.__version__, torchcodec.__version__)
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"
```

## The four JetPack-7 fixes (all applied)

### 1. lerobot forces the wrong CUDA index

`~/code/lerobot/pyproject.toml:368-375` declares:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

On JetPack 7 (CUDA 13) this yields `torchvision 0.26.0+cu128` paired with
`torch 2.11.0+cu130`. Symptom:

```
RuntimeError: operator torchvision::nms does not exist
```

**Fix:** reinstall both from PyPI afterwards with `--index-url https://pypi.org/simple`
(not `--extra-index-url`). Result: `torchvision 0.26.0+cu130`, matching ABI.

Note: this also leaves ~15 stale `*-cu12` NVIDIA packages in the venv (~3 GB).
Harmless — torch links only the cu13 libraries.

### 2. numpy gets dragged out of range

The forced torch reinstall resolves numpy freshly, without lerobot's constraint,
and installs 2.5.2. lerobot requires `numpy>=2.0.0,<2.3.0` (upper bound imposed
by `opencv-python-headless`).

**Fix:** `uv pip install "numpy<2.3"` → 2.2.6.

### 3. torchcodec needs NVIDIA NPP

torchcodec's `libtorchcodec_core*.so` link `libnppicc.so.13`, and torch does not
pull NPP in. Symptom: `OSError: Could not load this library: .../libtorchcodec_core4.so`.

**Fix:** `uv pip install nvidia-npp` → 13.1.2.81.

> Install `nvidia-npp`, **not** `nvidia-npp-cu13` — the latter is a 0.0.1 stub.
> (Naming is inconsistent upstream: `nvidia-cudnn-cu13` is real, `nvidia-npp` is not suffixed.)

### 4. The CUDA 13 libraries are not on the loader path

All CUDA 13 libraries live in `.venv/lib/python3.12/site-packages/nvidia/cu13/lib`,
which nothing adds to the dynamic loader's search path. torch resolves its own
copies internally; torchcodec's `dlopen` does not. Symptom:

```
OSError: libnvrtc.so.13: cannot open shared object file
```

**Fix:** `_cu13_preload.py` + `_cu13_preload.pth` in site-packages, which
`ctypes.CDLL(..., RTLD_GLOBAL)` every `.so` in that directory at interpreter
startup, registering the sonames process-wide.

> A plain `sitecustomize.py` **does not work** — Debian/Ubuntu ships
> `/usr/lib/python3.12/sitecustomize.py`, which shadows it. The `.pth` hook is
> required, and it covers both `python` and the `lerobot-*` console scripts.

> **The hook must skip `libnvblas.so`.** NVBLAS is a BLAS *interposer*: loaded
> `RTLD_GLOBAL` it captures BLAS symbols process-wide, and with no CPU fallback
> configured it segfaults the first time numpy or OpenCV call LAPACK. Found via
> `cv2.calibrateCamera` crashing with `[NVBLAS] CPU Blas library need to be
> provided` / exit 139. torchcodec needs `cudart`, `nvrtc` and `npp` only. The
> installed hook has `if "nvblas" in name: continue`; keep that if it is ever
> rewritten. Regression check:
> `./.venv/bin/python -c "import cv2,numpy as np; cv2.calibrateCamera(...)"`
> must not print any `[NVBLAS]` line.

## Housekeeping

- `.venv` is ~7.6 GB (includes the stale cu12 packages).
- `~/.cache/uv` grew to ~13 GB during setup; safe to prune with `uv cache clean`.
