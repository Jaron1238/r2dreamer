# R2Dreamer (FPV/Drone Fork)

This repository is a **project-specific fork** of R2-Dreamer.
In addition to the original world-model components, it includes a multi-phase workflow for FPV/drone use cases with streaming datasets, safety signals, and (from Phase 3) online interaction.

## What this project currently does

- **Hydra-based training** via `train.py`.
- **Offline training with Hugging Face streaming datasets** (`FPVDataset`), including optional OSD filtering.
- **Depth/raw-image path** for safety-related inputs (`raw_image`, `crash`).
- **Phase-based setup**:
  - **Phase 1**: World-model training (Actor/Value frozen).
  - **Phase 2**: Encoder/RSSM frozen, behavior cloning + safety learning.
  - **Phase 3**: Online training via `envs/drone_sim.py` (adapter placeholder).
  - **Phase 4**: Real deployment scaffold in `fly_real.py`.

## Project structure (key files)

- `train.py` – entry point (Hydra config, trainer selection, checkpoint load/save)
- `trainer.py` – `FPVDataset`, offline/online trainers, safety raw-image sources
- `dreamer.py` – core model with phase logic and safety head
- `configs/configs.yaml` – central project configuration
- `envs/drone_sim.py` – lightweight Gymnasium adapter for Phase 3
- `fly_real.py` – integration scaffold for real flights (MAVSDK pipeline)
- `runs/*.sh` – batch scripts for classic benchmarks (DMC, Atari, Crafter, ...)

## Setup

Tested with Ubuntu 24.04 and Python 3.11.

```bash
pip install -r requirements.txt
```

Optional (Docker): see [`docs/docker.md`](docs/docker.md).

Depth preprocessing works without `depth-anything-v2` (grayscale fallback is built in).
If you want the Depth Anything backend, install it separately in an environment that
uses a compatible PyTorch/TorchVision pair.

## Quickstart

### 1) Default run

```bash
python3 train.py logdir=./logdir/test
```

### 2) TensorBoard

```bash
tensorboard --logdir ./logdir
```

### 3) Resume from checkpoint

```bash
python3 train.py checkpoint=/path/to/latest.pt logdir=./logdir/resume
```

## Important configuration for this project

Core project settings are in `configs/configs.yaml`:

- `phase`: training phase (1/2/3)
- `use_depth`: enables depth preprocessing
- `dataset.mode`: typically `streaming`
- `dataset.hf_repo`: Hugging Face dataset repo (set this to your repo)
- `dataset.require_osd`: uses OSD samples (if available)
- `dataset.raw_image_mode`: `grayscale` or `rgb`
- `dataset.raw_dataset.*`: additional safety raw-image source (folders or streaming)
- `model.safety_input_key`: `image` or `raw_image`

Example:

```bash
python3 train.py \
  phase=2 \
  use_depth=true \
  dataset.hf_repo='your_user/your_fpv_dataset' \
  dataset.require_osd=true \
  model.safety_input_key=raw_image \
  logdir=./logdir/phase2_run
```

## Recommended phase workflow

1. **Phase 1 (World model pretraining)**
   - Focus on representation and dynamics learning.
2. **Phase 2 (Policy/safety adaptation)**
   - Uses safety signal (`crash`) and BC-like components.
3. **Phase 3 (Online simulation)**
   - Connect simulator/bridge through `DroneSimEnv`.
4. **Phase 4 (Real flight scaffold)**
   - `fly_real.py` as integration point for camera + MAVSDK + exported models.

## Classic R2-Dreamer benchmarks (optional)

Original benchmark scripts are still available, for example:

- `runs/dmc.sh`
- `runs/atari.sh`
- `runs/crafter.sh`
- `runs/memorymaze.sh`
- `runs/metaworld.sh`

Use these if you still need classic comparison experiments.

## Development / formatting

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Citation

If you use the R2-Dreamer base, please cite:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

Original paper: https://openreview.net/forum?id=Je2QqXrcQq
